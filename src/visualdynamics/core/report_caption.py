"""The caption a report plot of a spectrum carries, read off the record.

A spectrum in a report is evidence only when a reader who was not at
the test can say what it is. That takes a few lines of text under the
plot, and they are the same lines on every plot of a kind: which
channel, at which location and in which axis; which run, and which
stretch of it; how the spectrum was estimated; the line spacing it was
computed at; the window and the overlap of its frames; the degrees of
freedom of its average; and the requirement drawn on it. The line
spacing and the degrees of freedom are not formalities. A resonance's
peak height in an autospectrum depends on the spacing, and how far a
line may sit from the reference by chance depends on the degrees of
freedom, so without them no two plots can be compared and no excursion
can be judged.

Every one of those facts is already held by the record or the objects
around it, so none of them has to be typed. `record_caption` reads
them into a `RecordCaption`:

- the channel is the record's DOF and quantity, and the channel
  table's row for it gives the channel number and its comment; the
  DOF gives the node and the direction measured, and a geometry gives
  the global axis that direction is nearest;
- the estimate, the line spacing and the equivalent degrees of freedom
  are the spectrum's own (`Psd.statistical_dof`, stamped when the
  spectrum was averaged); a banded spectrum carries the line spacing of
  the narrowband spectrum it was banded from, and its degrees of
  freedom are those of the lines each band sums: Satterthwaite's count
  for a sum of correlated lines, each weighted by the power it puts in
  the band, with the window's correlation between lines
  (`Averaging.line_correlation`), the rule `chance` judges summed lines
  by;
- the window, the overlap, the frames and the stretch of the record
  are the `Averaging` the spectrum was made with, which a spectrum does
  not carry itself (`Project.averaging_of` gives it from the
  provenance);
- the requirement is the specification drawn with the spectrum: which
  limits it has, and the overall rms the record measured (and, where
  the comparison scales the measurement, the rms as drawn) beside the
  one the specification calls for, over the band it is written.

A shock response spectrum, a sine level and a time history carry
captions of their own kind (`caption_kind`). `srs_caption` gives an
SRS's Q, its spacing in lines per octave, the kind of peak, the filters
its run came through, the run's sample rate, and each record drawn
against the points rule in each polarity the project holds.
`sine_caption` gives a sine level's sweep as the requirement specifies
it, the detector it was extracted with and the distortion check already
made of its tone. `time_caption` gives a time history's span, the
stretch its averaging reads, its sample rate and each channel's range.
Each caption names its rows (`ROWS`) and the rows that belong to one
channel (`CHANNEL_ROWS`), so a grid states the others once.

A fact nothing records is written as "not recorded" rather than left
out, so a reader sees the gap instead of assuming the line was never
needed. `Project.caption_of` gathers the record's surroundings from a
project and returns the caption; the report draws it under a plot
block that asks for it, and `plot.plot_captioned` under a saved plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np

#: the rows of a caption, in the order they are read: what was
#: measured and where, what data, how it was analyzed, and against what
ROWS = ('Channel', 'Location and axis', 'Run', 'Estimate', 'Resolution',
        'Window', 'DOF', 'Requirement')

#: the rows that belong to one channel; the others are the settings a
#: figure of several channels shares, said once for all of them
CHANNEL_ROWS = ('Channel', 'Location and axis', 'Requirement')

#: what a fact nothing records is written as
NOT_RECORDED = 'not recorded'


@dataclass(frozen=True)
class CaptionChannel:
    """One channel a caption names.

    `dof` is the record's DOF string and `quantity` what it measures
    ('acceleration'). `channel` is the channel table's number for it,
    and `description` the table's comment (empty when there is none, or
    when it only repeats the DOF); both are None or empty without a
    table. `node` and `direction` are read from the DOF. `nearest_axis`
    is the signed global axis the measured direction is nearest on the
    geometry ('Z+') and `axis_angle` how far off it the direction sits,
    in degrees; both None without a geometry that places the channel.
    `against` is the reference channel of a cross term, read the same
    way; None for an autospectrum.
    """

    dof: str
    quantity: str
    channel: int | None = None
    node: int | None = None
    direction: str = ''
    description: str = ''
    nearest_axis: str | None = None
    axis_angle: float | None = None
    against: CaptionChannel | None = None

    def name(self) -> str:
        """'channel 3, 110Z+ acceleration', or the DOF and quantity
        alone when no channel table numbers it; a cross term names its
        reference after it, '1Z+ acceleration against 2Z+
        acceleration'."""
        what = f'{self.dof} {self.quantity}'.strip()
        own = what if self.channel is None else f'channel {self.channel}, {what}'
        if self.against is None:
            return own
        return f'{own} against {self.against.name()}'

    def place(self) -> str:
        """The location and the axis: 'node 110 (bracket), Z+ (nearest
        global Z+)', and a cross term's reference after it."""
        if self.against is not None:
            alone = replace(self, against=None).place()
            return f'{alone}; against {self.against.place()}'
        where = NOT_RECORDED if self.node is None else f'node {self.node}'
        if self.description:
            where += f' ({self.description})'
        axis = self.direction or 'direction not recorded'
        if self.nearest_axis is not None:
            # a direction measured in a local system reads in that
            # system; the global axis it is nearest is what a reader
            # finds on their own drawing
            off = ('' if self.axis_angle is None or self.axis_angle < 0.5
                   else f', {self.axis_angle:.0f}° off')
            axis += f' (nearest global {self.nearest_axis}{off})'
        return f'{where}, {axis}'


@dataclass(frozen=True)
class RecordCaption:
    """The facts a report plot of a spectrum carries, read off its
    record by `record_caption`.

    Attributes
    ----------
    channels : tuple of CaptionChannel
        The channels drawn, one per record captioned.
    kind : str
        'ASD' for an autospectrum, 'CPSD' for a cross term, 'ASD and
        CPSD' for a mixture.
    run : str or None
        The run the spectrum was computed from: the time history's name
        or the record's block label.
    stretch : tuple of float or None
        The (start, stop) of the stretch of the run averaged, in seconds
        on the run's clock.
    line_spacing : float or None
        The line spacing in Hz; None when the lines are not evenly
        spaced (and, for a banded spectrum, when the narrowband
        spectrum it came from is not known).
    per_octave : int or None
        Bands per octave of a banded spectrum, None for a narrowband one.
    window : str or None
        The window each frame was weighted by.
    overlap : float or None
        The overlap of successive frames, a fraction of a frame.
    frame_length : int or None
        The samples in a frame.
    frames : int or None
        Frames averaged in each record.
    pooled : tuple of int or None
        Records of each channel pooled into its average, as a capture
        saved one record per average is, one count per channel; the
        channels of one figure need not have the same.
    dof : tuple of float or None
        The equivalent statistical degrees of freedom of each channel's
        average, discounted for overlap; for a banded spectrum, of each
        narrowband line it was banded from.
    band_dof : tuple of float or None
        For a banded spectrum, the (lowest, highest) equivalent degrees
        of freedom of its bands over the records captioned
        (`band_dof`); None for a narrowband one, or where the
        narrowband spectrum, its averaging or a record's autospectrum
        is not known.
    nominal_dof : tuple of int or None
        Two per frame averaged, the count without the discount.
    requirement : str or None
        The name of the specification drawn, None when none is.
    limits : tuple of str
        Which of the specification's limit pairs are drawn: 'warning',
        'abort', or both.
    measured_rms, reference_rms : float or None
        The overall rms of the one record captioned, as measured, and
        the rms the specification calls for, both over the band the
        comparison judged (`compliance.compare`), in `rms_unit`.
    drawn_rms : float or None
        The measured rms scaled as it is drawn and judged, where the
        comparison scales it (`scale_db` not zero); None otherwise.
    rms_unit : str
        The display unit of the two rms values.
    rms_band : tuple of float or None
        The (low, high) band in Hz the two rms values are taken over.
    scale_db : int
        The decibels the measured curve is drawn scaled by against the
        specification (`compliance.comparison_scale_db`); 0 for none.
    averaging : Averaging or None
        The averaging the spectrum was made with, whose `describe` is
        the window row.
    not_compared : str or None
        Why the one record captioned has no rms against the
        specification drawn, in words (`compliance.compare`'s refusal of
        another quantity, no finite line in the band, or no record of
        the specification for its channel); None when it has them or
        when no specification is drawn.
    """

    #: the rows, and those that belong to one channel, as every
    #: caption kind names them
    ROWS: ClassVar[tuple[str, ...]] = ROWS
    CHANNEL_ROWS: ClassVar[tuple[str, ...]] = CHANNEL_ROWS

    channels: tuple[CaptionChannel, ...]
    kind: str
    run: str | None = None
    stretch: tuple[float, float] | None = None
    line_spacing: float | None = None
    per_octave: int | None = None
    window: str | None = None
    overlap: float | None = None
    frame_length: int | None = None
    frames: int | None = None
    pooled: tuple[int, ...] | None = None
    dof: tuple[float, ...] | None = None
    nominal_dof: tuple[int, ...] | None = None
    band_dof: tuple[float, float] | None = None
    requirement: str | None = None
    limits: tuple[str, ...] = ()
    measured_rms: float | None = None
    reference_rms: float | None = None
    rms_unit: str = ''
    rms_band: tuple[float, float] | None = None
    scale_db: int = 0
    drawn_rms: float | None = None
    averaging: Any = None
    not_compared: str | None = None

    def rows(self) -> list[tuple[str, str]]:
        """The caption as (label, text) rows, one per `ROWS` entry, in
        that order.

        Returns
        -------
        list of tuple of str
            Each row's label and what it says.
        """
        return list(zip(self.ROWS, (self._channel(), self._place(), self._run(),
                               self._estimate(), self._resolution(),
                               self._window(), self._dof(),
                               self._requirement()), strict=True))

    def text(self) -> str:
        """The caption as one paragraph: each row as 'Label: text', the
        rows joined by '. '.

        Returns
        -------
        str
        """
        return '. '.join(f'{label}: {said}' for label, said in self.rows()) + '.'

    # ---- one row each --------------------------------------------------

    def _channel(self) -> str:
        return '; '.join(channel.name() for channel in self.channels)

    def _place(self) -> str:
        return '; '.join(channel.place() for channel in self.channels)

    def _run(self) -> str:
        if self.run is None:
            return NOT_RECORDED
        if self.stretch is None:
            return self.run
        start, stop = self.stretch
        return f'{self.run}, {start:.4g} s to {stop:.4g} s'

    def _estimate(self) -> str:
        count = len(self.channels)
        said = (f'{self.kind}, single channel' if count == 1
                else f'{self.kind} of each of {count} channels')
        if self.frames is None:
            said += ', averaging not recorded'
        else:
            # TimeHistory.compute_psds takes the plain mean of the
            # frames' powers: a linear average, never an exponential or
            # a peak hold
            said += (f', linear average of {self.frames} '
                     f'frame{"s" * (self.frames != 1)}')
            said += self._pooling()
        if self.per_octave is not None:
            said += f', banded to 1/{self.per_octave} octave'
        return said

    def _pooling(self) -> str:
        """How many records each channel pooled, when any pooled more
        than one: one count when every channel has the same, else each
        channel's own, since the counts set each channel's DOF."""
        if not self.pooled or max(self.pooled) <= 1:
            return ''
        if len(set(self.pooled)) == 1:
            return f' in each of {self.pooled[0]} records'
        each = ', '.join(
            f'{count} record{"s" * (count != 1)} at {channel.dof}'
            for channel, count in zip(self.channels, self.pooled,
                                      strict=True))
        return f' in each record, pooled over {each}'

    def _resolution(self) -> str:
        lines = (None if self.line_spacing is None
                 else f'{self.line_spacing:.4g} Hz lines')
        if self.per_octave is not None:
            bands = f'1/{self.per_octave}-octave bands'
            return bands if lines is None else f'{bands}, from {lines}'
        return lines or 'lines not evenly spaced'

    def _window(self) -> str:
        # the averaging's own statement, the one every caption, label
        # and summary uses; no second wording of it here
        if self.averaging is None:
            return NOT_RECORDED
        return self.averaging.describe()

    def _dof(self) -> str:
        if not self.dof:
            return NOT_RECORDED
        said = _span(self.dof) + ' equivalent'
        if self.nominal_dof:
            said += f' ({_span(self.nominal_dof)} nominal)'
        if self.per_octave is None:
            return said
        if self.band_dof is not None:
            # a band holds the lines it sums, and their degrees of
            # freedom with them; the band's count is the one its scatter
            # follows
            return (f'{_span(self.band_dof)} equivalent per band, from '
                    f'{said} per narrowband line')
        # without the lines' powers and the window's correlation the
        # bands' count cannot be worked out; the lines' is what is known
        return said + ' per narrowband line, before banding'

    def _requirement(self) -> str:
        if self.requirement is None:
            return 'none drawn'
        drawn = ' and '.join(self.limits)
        said = (f'{self.requirement}, reference with its {drawn} limits drawn'
                if drawn else f'{self.requirement}, reference drawn')
        if self.measured_rms is not None and self.reference_rms is not None:
            # the measured value as measured and, where the comparison
            # scales it, as drawn; the scale itself is in the figure's
            # legend and its caption, and is not said a third time
            unit = f' {self.rms_unit}' if self.rms_unit else ''
            said += f'; {self.measured_rms:.3g}{unit} rms measured'
            if self.drawn_rms is not None:
                said += f', {self.drawn_rms:.3g}{unit} rms as drawn,'
            said += f' against {self.reference_rms:.3g}{unit} rms specified'
            if self.rms_band is not None:
                said += (f', {self.rms_band[0]:.4g} Hz to '
                         f'{self.rms_band[1]:.4g} Hz')
        elif self.not_compared is not None:
            said += f'; not compared: {self.not_compared}'
        return said


def _span(values: Sequence[float]) -> str:
    """One number when every channel has the same, else 'low to high'."""
    low, high = float(min(values)), float(max(values))
    if np.isclose(low, high, rtol=1e-9, atol=0.0):
        return f'{low:.4g}'
    return f'{low:.4g} to {high:.4g}'


def band_dof(frequencies: Any, values: Any, dof: float, correlation: Any,
             left: Any, right: Any) -> np.ndarray:
    """The equivalent degrees of freedom of each band of a banded
    autospectrum, from the narrowband lines it sums.

    Each line counts in a band with the power it puts there: its value
    times the part of its bin inside the band (`octave.resample`'s
    rule). Lines `dof` each whose estimates vary together by
    `correlation` at 0, 1, 2, ... lines apart (`Averaging.line_correlation`)
    sum to Satterthwaite's count, the rule `chance` judges summed lines
    by: N times `dof` for N equal, independent lines, fewer under any
    window but a rectangle.

    Parameters
    ----------
    frequencies, values : array_like
        The narrowband lines, evenly spaced, and the autospectrum on
        them.
    dof : float
        The equivalent degrees of freedom of one line.
    correlation : array_like
        The correlation between line estimates, starting at 1.
    left, right : array_like
        The bands' edges in Hz (`octave.bin_bounds`).

    Returns
    -------
    numpy.ndarray
        One count per band; NaN for a band no line with power reaches.
    """
    from .chance import _sum_dof
    from .octave import bin_bounds

    frequencies = np.asarray(frequencies, dtype=float)
    power = np.real(np.asarray(values, dtype=complex))
    power = np.where(np.isfinite(power) & (power > 0.0), power, 0.0)
    low, high = bin_bounds(frequencies)
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    share = np.clip(np.minimum(high[None, :], right[:, None])
                    - np.maximum(low[None, :], left[:, None]), 0.0, None)
    weights = share * power[None, :]
    correlation = np.asarray(correlation, dtype=float)
    out = np.full(len(left), np.nan)
    for k, row in enumerate(weights):
        held = np.flatnonzero(row > 0.0)
        if held.size:
            out[k] = _sum_dof(row[held[0]:held[-1] + 1], float(dof),
                              correlation)
    return out


def _spacing(abscissa: np.ndarray) -> float | None:
    """The line spacing of an evenly spaced axis
    (`tone_readout.line_spacing`, the one test of even lines), else
    None."""
    from .tone_readout import line_spacing

    try:
        return line_spacing(abscissa)
    except ValueError:
        return None


def _per_octave(abscissa: np.ndarray) -> int | None:
    """Bands per octave of a banded axis, from the ratio of its first
    two centers."""
    f = np.asarray(abscissa, dtype=np.float64)
    if f.size < 2 or f[0] <= 0.0 or f[1] <= f[0]:
        return None
    return max(round(1.0 / np.log2(f[1] / f[0])), 1)


def caption_channel(spectrum: Any, record: int, channel_table: Any = None,
                    geometry: Any = None) -> CaptionChannel:
    """The channel one record of a spectrum is measured at, with what a
    channel table and a geometry add about it.

    The table's row is the one whose DOF and channel type match the
    record's; a row of the right DOF whose type is blank answers when
    no row matches both, since a drive point's force channel shares its
    DOF with the acceleration channel beside it and only the type tells
    them apart.

    Parameters
    ----------
    spectrum : DataArray
        The spectrum.
    record : int
        Which record.
    channel_table : ChannelTable, optional
        The test's channel table.
    geometry : Geometry, optional
        The geometry the channel is measured on.

    Returns
    -------
    CaptionChannel
    """
    from .data import channel_quantities

    response, reference = (str(dof) for dof in spectrum.record_pair(record))
    dimension = str(spectrum.known_dim(record))
    out, into = (('', '') if dimension == 'unknown'
                 else channel_quantities(dimension))
    own = _channel_at(response, out, channel_table, geometry)
    if response == reference and out == into:
        return own
    # a cross term is of two channels, and names both: by its response
    # alone, a CPSD's 1Z+ against 2Z+ read as 1Z+ a second time
    return replace(own, against=_channel_at(reference, into, channel_table,
                                            geometry))


def _channel_at(dof: str, quantity: str, channel_table: Any,
                geometry: Any) -> CaptionChannel:
    """One channel by its DOF and quantity, with its channel table row
    and the global axis nearest its direction."""
    from .data import parse_dof

    node, direction = parse_dof(dof)
    number, description = None, ''
    nearest, angle = None, None
    if channel_table is not None:
        row = _table_row(channel_table, dof, quantity)
        if row is not None:
            frame = channel_table.frame
            number = int(frame['channel'].iloc[row])
            comment = str(frame['comment'].iloc[row]).strip()
            # a controller often writes the DOF itself as the comment,
            # which would only say the channel's name twice
            description = '' if comment == dof else comment
            if geometry is not None:
                _vector, nearest, angle = channel_table.orientation(row, geometry)
    return CaptionChannel(dof=dof, quantity=quantity, channel=number,
                          node=node, direction=direction,
                          description=description, nearest_axis=nearest,
                          axis_angle=angle)


def _table_row(table: Any, dof: str, quantity: str) -> int | None:
    """The channel table row for a DOF and a quantity, or None."""
    dofs = table.dof_strings()
    types = list(table.frame['channel_type'])
    same = [row for row, name in enumerate(dofs) if name == dof]
    exact = [row for row in same if types[row] == quantity]
    if exact:
        return exact[0]
    blank = [row for row in same if not types[row]]
    if blank:
        return blank[0]
    # one row of that DOF and nothing to tell it apart by: the
    # record's quantity is unknown, so the row is the only candidate
    return same[0] if len(same) == 1 and not quantity else None


def record_caption(spectrum: Any, records: int | Sequence[int] = 0, *,
                   averaging: Any = None, run: str | None = None,
                   sample_rate: float | None = None, origin: float = 0.0,
                   channel_table: Any = None, geometry: Any = None,
                   narrowband: Any = None, per_octave: int | None = None,
                   specification: Any = None, requirement: str | None = None,
                   scale_db: int | None = None,
                   unit_system: Any = None) -> RecordCaption:
    """The caption a report plot of a spectrum carries, read off the
    record and the objects around it.

        caption = record_caption(psds, 2, averaging=averaging,
                                 run='Run 6', channel_table=table)
        caption.text()

    Nothing in it is typed: each row is read from the record, its
    `averaging`, the channel table, the geometry and the specification.
    Whatever is not given is written as "not recorded" in its row.
    `Project.caption_of` gathers all of these from a project.

    Parameters
    ----------
    spectrum : Psd
        The measured spectrum: an autospectrum or a cross-spectral
        density, narrowband or banded. A specification is the
        requirement, not a measurement, and is refused.
    records : int or sequence of int, default 0
        The records the plot draws; the caption names each channel and
        states the settings they share.
    averaging : Averaging, optional
        The averaging the spectrum was made with (for a banded
        spectrum, the narrowband one's). It gives the window, the
        overlap, the frames and the nominal degrees of freedom.
    run : str, optional
        The run the spectrum was computed from. Defaults to the
        record's block label, when the records have one.
    sample_rate : float, optional
        The run's sample rate in Hz; with `averaging`, it gives the
        stretch of the run averaged.
    origin : float, default 0.0
        The run's first instant in seconds, for a run whose clock does
        not begin at zero.
    channel_table : ChannelTable, optional
        The test's channel table: the channel number and comment.
    geometry : Geometry, optional
        The geometry the channels are measured on: the global axis each
        measured direction is nearest.
    narrowband : Psd, optional
        For a banded spectrum, the narrowband spectrum it was banded
        from, record for record: its line spacing and degrees of
        freedom.
    per_octave : int, optional
        Bands per octave of a banded spectrum; read from its band
        centers when omitted.
    specification : Specification, optional
        The requirement drawn with the spectrum.
    requirement : str, optional
        The specification's name; 'specification' when omitted.
    scale_db : int, optional
        The decibels the measured curve is drawn scaled by; resolved
        the way every comparison resolves it when omitted
        (`compliance.comparison_scale_db`).
    unit_system : UnitSystem, optional
        The display units the rms values are written in; the default
        system when omitted.

    Returns
    -------
    RecordCaption

    Raises
    ------
    TypeError
        If `spectrum` is not a measured `Psd`.
    IndexError
        If a record is not one of the spectrum's.
    """
    from ..units import DEFAULT_SYSTEM
    from .data import Bounded, Psd

    if not isinstance(spectrum, Psd) or isinstance(spectrum, Bounded):
        raise TypeError('a report caption reads a measured spectrum (a Psd); '
                        f'{type(spectrum).__name__} is not one')
    wanted = [int(records)] if np.ndim(records) == 0 else [int(r) for r in records]
    if not wanted:
        raise IndexError('a caption needs at least one record')
    for record in wanted:
        if not 0 <= record < spectrum.num_records:
            raise IndexError(f'record {record} is not one of the '
                             f'{spectrum.num_records} records')
    us = DEFAULT_SYSTEM if unit_system is None else unit_system

    channels = tuple(caption_channel(spectrum, record, channel_table, geometry)
                     for record in wanted)
    # one DOF named twice over two quantities, a drive point's force
    # against its acceleration, is a cross term too
    autos = [spectrum.is_autospectrum(record) for record in wanted]
    kind = 'ASD' if all(autos) else 'CPSD' if not any(autos) else 'ASD and CPSD'

    banded = getattr(spectrum, 'bandwidth', None) is not None
    if banded and narrowband is not None \
            and narrowband.num_records != spectrum.num_records:
        narrowband = None           # not this spectrum's source, record for record
    if banded:
        per_octave = per_octave or _per_octave(spectrum.abscissa)
        lines_of = narrowband
    else:
        per_octave = None
        lines_of = spectrum
    spacing = None if lines_of is None else _spacing(lines_of.abscissa)

    stamped = None if lines_of is None else getattr(lines_of, 'statistical_dof', None)
    dof = (None if stamped is None
           else tuple(float(stamped[record]) for record in wanted))
    nominal, pooled = None, None
    if dof is not None and averaging is not None:
        # the stamp is the count over every record of the channel
        # pooled; one record's frames give the share each adds, so the
        # pooled count, and with it the nominal 2n, follows
        one = averaging.statistical_dof(records=1)
        counts = [max(round(each / one.equivalent), 1) for each in dof]
        nominal = tuple(one.nominal * count for count in counts)
        pooled = tuple(counts)
    bands = None
    if banded and dof is not None and averaging is not None \
            and all(narrowband.is_autospectrum(record) for record in wanted):
        from .octave import bin_bounds

        left, right = bin_bounds(spectrum.abscissa, spectrum.bandwidth)
        correlation = averaging.line_correlation()
        counts = np.concatenate([
            band_dof(narrowband.abscissa, narrowband.ordinate[record], each,
                     correlation, left, right)
            for record, each in zip(wanted, dof)])
        counts = counts[np.isfinite(counts)]
        if counts.size:
            bands = (float(counts.min()), float(counts.max()))

    if run is None and spectrum.block is not None:
        labels = {spectrum.block[record] for record in wanted}
        run = ', '.join(sorted(labels)) or None
    stretch = None
    if averaging is not None and sample_rate:
        stretch = (origin + averaging.start_sample(sample_rate) / sample_rate,
                   averaging.stop(sample_rate, origin))

    limits: tuple[str, ...] = ()
    measured_rms = reference_rms = drawn_rms = not_compared = None
    band: tuple[float, float] | None = None
    unit = ''
    if specification is not None:
        from .compliance import (
            compare,
            comparison_scale_db,
            matched_records,
        )
        from .data import channel_quantities

        written = {name for name, values in (specification.limits or {}).items()
                   if values is not None}
        limits = tuple(kind_ for kind_ in ('warning', 'abort')
                       if {f'{kind_}_lower', f'{kind_}_upper'} & written)
        if scale_db is None:
            scale_db = comparison_scale_db(specification, spectrum)
        if len(wanted) == 1:
            pairs = matched_records(specification, spectrum,
                                    measured_records=wanted)
            if pairs:
                # the comparison the verdict table makes, so the two
                # numbers here are the table's: both over the band it
                # judged, the measurement scaled as it is drawn
                _label, si, mi = pairs[0]
                compared = compare(specification, spectrum, si, mi, scale_db)
                if not compared.get('lines'):
                    finite = np.isfinite(np.real(spectrum.ordinate[mi]))
                    not_compared = compared.get('refused') or (
                        "none of the record's lines is a finite number"
                        if not finite.any() else
                        'no line of the record and the specification is '
                        'finite in the band they share')
                else:
                    band = tuple(float(edge) for edge in compared['band'])
                    dimension = str(spectrum.known_dim(mi))
                    reference_rms = float(compared['specification_rms'])
                    # as measured, and as drawn where the comparison
                    # scales the measurement: a reader holding the raw
                    # data reads the first, the figure shows the second
                    measured_rms = float(
                        compare(specification, spectrum, si, mi,
                                0)['measured_rms'] if scale_db
                        else compared['measured_rms'])
                    if scale_db:
                        drawn_rms = float(compared['measured_rms'])
                    if dimension != 'unknown':
                        amplitude = channel_quantities(dimension)[0]
                        measured_rms = float(us.from_si(measured_rms, amplitude))
                        reference_rms = float(us.from_si(reference_rms,
                                                         amplitude))
                        if drawn_rms is not None:
                            drawn_rms = float(us.from_si(drawn_rms,
                                                         amplitude))
                        unit = us.label_text(amplitude)
            else:
                not_compared = ('the specification holds no record for '
                                f'{channels[0].name()}')
    return RecordCaption(
        channels=channels, kind=kind, run=run, stretch=stretch,
        line_spacing=spacing, per_octave=per_octave,
        window=None if averaging is None else averaging.window,
        overlap=None if averaging is None else averaging.overlap,
        frame_length=None if averaging is None else averaging.frame_length,
        frames=None if averaging is None else averaging.frames,
        pooled=pooled, dof=dof, nominal_dof=nominal, band_dof=bands,
        requirement=(None if specification is None
                     else requirement or 'specification'),
        limits=limits, measured_rms=measured_rms,
        reference_rms=reference_rms, rms_unit=unit, rms_band=band,
        scale_db=int(scale_db or 0) if specification is not None else 0,
        drawn_rms=drawn_rms, averaging=averaging, not_compared=not_compared)


# ---- the shock response spectrum, the sine level, the time history -------


def caption_kind(data: Any) -> str | None:
    """Which caption a record carries: 'spectrum' for a measured `Psd`,
    'srs' for a measured `Srs`, 'sine' for a `SineLevel` or a
    `SineLevelSet`, 'time' for a measured `TimeHistory`; None for
    anything else, a requirement among them, since a requirement is
    what a measurement is read against and not a measurement.

    Parameters
    ----------
    data : object
        The object a figure draws.

    Returns
    -------
    str or None
    """
    from .data import Bounded, Psd, Srs, TimeHistory, TransientSpecification
    from .sine import SineLevel, SineLevelSet

    if isinstance(data, Bounded):
        return None
    if isinstance(data, Psd):
        return 'spectrum'
    if isinstance(data, Srs):
        return 'srs'
    if isinstance(data, (SineLevel, SineLevelSet)):
        return 'sine'
    if isinstance(data, TimeHistory) \
            and not isinstance(data, TransientSpecification):
        return 'time'
    return None


def _records_wanted(data: Any, records: int | Sequence[int]) -> list[int]:
    """The records asked for, each checked against `data`'s count."""
    wanted = [int(records)] if np.ndim(records) == 0 else [int(r) for r in records]
    if not wanted:
        raise IndexError('a caption needs at least one record')
    for record in wanted:
        if not 0 <= record < data.num_records:
            raise IndexError(f'record {record} is not one of the '
                             f'{data.num_records} records')
    return wanted


def _one_channel(data: Any, record: int, channel_table: Any,
                 geometry: Any) -> CaptionChannel:
    """The channel one record of a single-quantity record measures."""
    from .data import channel_quantities

    dimension = str(data.known_dim(record))
    quantity = ('' if dimension == 'unknown'
                else channel_quantities(dimension)[0])
    return _channel_at(str(data.response_dof[record]), quantity,
                       channel_table, geometry)


def _channels(data: Any, wanted: Sequence[int], channel_table: Any,
              geometry: Any) -> tuple[CaptionChannel, ...]:
    """Each channel the records drawn measure, once, in order: an SRS
    holds one record per channel per event."""
    out: dict[tuple[str, str], CaptionChannel] = {}
    for record in wanted:
        key = (str(data.response_dof[record]), str(data.known_dim(record)))
        if key not in out:
            out[key] = _one_channel(data, record, channel_table, geometry)
    return tuple(out.values())


def _events(data: Any, wanted: Sequence[int]) -> tuple[str, ...]:
    """The block labels of the records drawn, in order, each once."""
    if data.block is None:
        return ()
    out: list[str] = []
    for record in wanted:
        label = data.block[record]
        if label and str(label) not in out:
            out.append(str(label))
    return tuple(out)


def lines_per_octave(abscissa: Any) -> float | None:
    """The lines per octave of a geometrically spaced axis, as an SRS's
    natural frequencies are (`srs.octave_frequencies`), else None: for
    an axis of fewer than two lines, one that does not rise from above
    zero, or one whose steps are not one ratio.

    Parameters
    ----------
    abscissa : array_like
        The natural frequencies in Hz.

    Returns
    -------
    float or None
    """
    f = np.asarray(abscissa, dtype=np.float64)
    if f.size < 2 or not np.all(np.isfinite(f)) or not f[0] > 0.0:
        return None
    ratios = f[1:] / f[:-1]
    if not np.all(ratios > 1.0) or not np.allclose(ratios, ratios[0],
                                                   rtol=1e-6, atol=0.0):
        return None
    return float(1.0 / np.log2(ratios[0]))


@dataclass(frozen=True)
class PointsCount:
    """One measured SRS record against the points rule, as
    `compliance.srs_tolerance` judged it: its polarity, its event, the
    points judged, how many lie inside the inner band and inside the
    outer one (None without an outer band), whether it passed (None when
    nothing was judged), why no point was judged when none was
    (`reason`), and how many of its points are not a number, which no
    rule judges (`missing`)."""

    kind: str
    event: str | None
    points: int
    inside: int
    inside_outer: int | None
    passed: bool | None
    reason: str | None
    missing: int

    def text(self) -> str:
        """'maximax, shock 1: 40 of 44 points inside the inner band and
        44 inside the outer, passes', or why nothing was judged."""
        where = self.kind + (f', {self.event}' if self.event else '')
        if self.reason is not None:
            return f'{where}: not judged, {self.reason}'
        said = (f'{where}: {self.inside} of {self.points} '
                f'point{"s" * (self.points != 1)} inside the inner band')
        if self.inside_outer is not None:
            said += f' and {self.inside_outer} inside the outer'
        if self.missing:
            said += (f' ({self.missing} point{"s" * (self.missing != 1)} '
                     'not a number, not judged)')
        return f'{said}, {"passes" if self.passed else "fails"}'


@dataclass(frozen=True)
class SrsCaption:
    """The facts a report plot of a shock response spectrum carries,
    read off its records by `srs_caption`.

    Attributes
    ----------
    channels : tuple of CaptionChannel
        The channels drawn, each once.
    run : str or None
        The time history the spectrum was computed from.
    events : tuple of str
        The events of the records drawn, by their block labels.
    q : float
        The oscillators' amplification.
    kind : str
        Which peak each point is: 'maximax', 'positive' or 'negative'.
    per_octave : float or None
        Natural-frequency lines per octave; None when they are not
        geometrically spaced.
    band : tuple of float or None
        The first and last natural frequency, in Hz.
    filters : tuple of str or None
        Each filter applied in the project on the way to the run, in
        the order applied; empty for none, None when the run is not
        known.
    sample_rate : float or None
        The run's sample rate in Hz.
    requirement : str or None
        The name of the requirement drawn, None when none is.
    rule : str
        The points rule in words, with the band it judges against.
    points : tuple of PointsCount
        Each record drawn against the requirement, in each polarity the
        project holds, the one drawn first.
    not_compared : str or None
        Why no record could be read against the requirement drawn.
    """

    #: the rows of an SRS caption, in the order they are read
    ROWS: ClassVar[tuple[str, ...]] = (
        'Channel', 'Location and axis', 'Run', 'Q', 'Spacing', 'Type',
        'Filter', 'Sample rate', 'Points rule')
    #: the rows that belong to one channel
    CHANNEL_ROWS: ClassVar[tuple[str, ...]] = (
        'Channel', 'Location and axis', 'Points rule')

    channels: tuple[CaptionChannel, ...]
    run: str | None = None
    events: tuple[str, ...] = ()
    q: float = 10.0
    kind: str = 'maximax'
    per_octave: float | None = None
    band: tuple[float, float] | None = None
    filters: tuple[str, ...] | None = None
    sample_rate: float | None = None
    requirement: str | None = None
    rule: str = ''
    points: tuple[PointsCount, ...] = ()
    not_compared: str | None = None

    def rows(self) -> list[tuple[str, str]]:
        """The caption as (label, text) rows, one per `ROWS` entry.

        Returns
        -------
        list of tuple of str
        """
        from .srs import damping_for

        run = NOT_RECORDED if self.run is None else self.run
        if self.events:
            run += ', ' + ', '.join(self.events)
        spacing = ('lines not geometrically spaced' if self.per_octave is None
                   else f'{self.per_octave:.3g} lines per octave')
        if self.band is not None:
            spacing += f', {self.band[0]:.4g} Hz to {self.band[1]:.4g} Hz'
        if self.filters is None:
            filtered = NOT_RECORDED
        else:
            filtered = ('; then '.join(self.filters) if self.filters
                        else 'none applied in the project')
        rate = (NOT_RECORDED if self.sample_rate is None
                else f'{self.sample_rate:.6g} Hz')
        q = (f'{self.q:g} ({100.0 * damping_for(self.q):.3g} % of critical '
             'damping)')
        return list(zip(self.ROWS, (
            _names(self.channels), _places(self.channels), run, q, spacing,
            self.kind, filtered, rate, self._points()), strict=True))

    def text(self) -> str:
        """The caption as one paragraph, 'Label: text' rows joined by
        '. '.

        Returns
        -------
        str
        """
        return _paragraph(self.rows())

    def _points(self) -> str:
        if self.requirement is None:
            return 'no requirement drawn'
        if not self.points:
            return f'{self.requirement}, not judged: {self.not_compared}'
        head = (f'{self.requirement}, {self.rule}' if self.rule
                else self.requirement)
        return f'{head}; ' + '; '.join(point.text() for point in self.points)


def _names(channels: Sequence[CaptionChannel]) -> str:
    return '; '.join(channel.name() for channel in channels)


def _places(channels: Sequence[CaptionChannel]) -> str:
    return '; '.join(channel.place() for channel in channels)


def _paragraph(rows: Sequence[tuple[str, str]]) -> str:
    return '. '.join(f'{label}: {said}' for label, said in rows) + '.'


def _unjudged(values: np.ndarray) -> str:
    """Why an SRS record had no point to judge, in words."""
    finite = np.isfinite(values)
    if not finite.any():
        return "none of the record's points is a number"
    if not (values[finite] > 0.0).any():
        return 'every point of the record is zero'
    return "no point of the record lies where the requirement is written"


def srs_caption(srs: Any, records: int | Sequence[int] = 0, *,
                run: str | None = None, sample_rate: float | None = None,
                filters: Sequence[str] | None = None,
                channel_table: Any = None, geometry: Any = None,
                specification: Any = None, requirement: str | None = None,
                polarities: Sequence[Any] = ()) -> SrsCaption:
    """The caption a report plot of a shock response spectrum carries,
    read off the records drawn and the objects around them.

        caption = srs_caption(srs, [0, 1], run='Drop 3',
                              specification=spec)
        caption.text()

    The Q, the kind of peak and the natural frequencies are the
    spectrum's own; the run, its sample rate and the filters applied on
    the way to it are what `Project.caption_of` reads from the
    provenance. Against a requirement each record drawn is judged by
    the points rule as the SRS points check judges it
    (`compliance.srs_tolerance`, paired as `compliance.srs_tolerance_all`
    pairs, its bands read from the requirement's limit curves where it
    has them), once in each polarity given: the spectrum drawn, then
    each of `polarities`, a spectrum of the same events at the same Q
    holding another peak, record for record.

    Parameters
    ----------
    srs : Srs
        The measured spectrum; a requirement is refused.
    records : int or sequence of int, default 0
        The records the plot draws.
    run : str, optional
        The time history the spectrum was computed from.
    sample_rate : float, optional
        The run's sample rate in Hz.
    filters : sequence of str, optional
        Each filter applied on the way to the run, in words; empty for
        none, None when it is not known.
    channel_table : ChannelTable, optional
        The channel number and comment.
    geometry : Geometry, optional
        The global axis each measured direction is nearest.
    specification : ShockSpecification, optional
        The requirement drawn.
    requirement : str, optional
        Its name; 'specification' when omitted.
    polarities : sequence of Srs, optional
        The same records in other polarities; one with another count of
        records is not the same records, and is passed over.

    Returns
    -------
    SrsCaption

    Raises
    ------
    TypeError
        If `srs` is not a measured `Srs`.
    IndexError
        If a record is not one of the spectrum's, or none is asked for.
    """
    from .compliance import _srs_pairs, srs_band_text, srs_tolerance

    if caption_kind(srs) != 'srs':
        raise TypeError('an SRS caption reads a measured shock response '
                        f'spectrum; {type(srs).__name__} is not one')
    wanted = _records_wanted(srs, records)
    abscissa = np.asarray(srs.abscissa, dtype=float)
    band = ((float(abscissa[0]), float(abscissa[-1])) if abscissa.size
            else None)

    points: list[PointsCount] = []
    rule = ''
    not_compared = None
    if specification is not None:
        spectra = [srs] + [other for other in polarities
                           if other.num_records == srs.num_records]
        for spectrum in spectra:
            pairs = {i: j for _dof, j, i, _event
                     in _srs_pairs(specification, spectrum)}
            for record in wanted:
                if record not in pairs:
                    continue
                result = srs_tolerance(specification, spectrum,
                                       pairs[record], record)
                event = (None if spectrum.block is None
                         else spectrum.block[record] or None)
                values = np.real(np.asarray(spectrum.ordinate[record],
                                            dtype=complex))
                reason = result.get('refused')
                if reason is None and not result['points']:
                    reason = _unjudged(values)
                judged = int(result['points'])
                points.append(PointsCount(
                    str(spectrum.kind), event, judged,
                    int(result.get('inside_count', 0)),
                    (None if result.get('outer') is None
                     else judged - int(result['outer_count'])),
                    result['passed'], reason,
                    int((~np.isfinite(values)).sum())))
                if not rule and reason is None:
                    share = (f'at least {result["percent"]:g} % of the '
                             'points inside the inner band'
                             if result['rule'] == 'share' else
                             'every point at or above the lower edge')
                    rule = ('band '
                            + srs_band_text(result['inner'], result['outer'])
                            + f', rule {share}')
                    if result.get('limits'):
                        rule += ", the band read from the requirement's limits"
        if not points:
            not_compared = 'the requirement names none of the channels drawn'
    return SrsCaption(
        channels=_channels(srs, wanted, channel_table, geometry), run=run,
        events=_events(srs, wanted), q=float(srs.q), kind=str(srs.kind),
        per_octave=lines_per_octave(abscissa), band=band,
        filters=None if filters is None else tuple(filters),
        sample_rate=None if sample_rate is None else float(sample_rate),
        requirement=(None if specification is None
                     else requirement or 'specification'),
        rule=rule, points=tuple(points), not_compared=not_compared)


@dataclass(frozen=True)
class SineCaption:
    """The facts a report plot of a sine sweep's extracted level
    carries, read off its records by `sine_caption`.

    Attributes
    ----------
    channels : tuple of CaptionChannel
        The control channels drawn.
    tone : str
        The tone the level was extracted for.
    run : str or None
        The time history the level was extracted from.
    onset : float or None
        Seconds into the run where the tone's sweep was found.
    sweep : tuple of str or None
        Each segment of the tone as specified, in words: its law, its
        rate and units, its direction and its span, or a dwell's
        frequency and duration; None without the requirement.
    drift_hz : float
        The sweep-clock correction the extraction applied at the end of
        the tone, in Hz; 0 for none.
    cycles : float or None
        The Vold-Kalman smoothing the level was read with, in cycles of
        the tone's own frequency; None when it is not recorded.
    distortion : object
        The distortion check of this tone (`sine_distortion.check_distortion`),
        or a reason in words why it could not be judged, or None when no
        check was made.
    """

    #: the rows of a sine caption, in the order they are read
    ROWS: ClassVar[tuple[str, ...]] = (
        'Channel', 'Location and axis', 'Run', 'Sweep', 'Detector',
        'Distortion')
    #: the rows that belong to one channel
    CHANNEL_ROWS: ClassVar[tuple[str, ...]] = (
        'Channel', 'Location and axis', 'Distortion')

    channels: tuple[CaptionChannel, ...]
    tone: str = ''
    run: str | None = None
    onset: float | None = None
    sweep: tuple[str, ...] | None = None
    drift_hz: float = 0.0
    cycles: float | None = None
    distortion: Any = None

    def rows(self) -> list[tuple[str, str]]:
        """The caption as (label, text) rows, one per `ROWS` entry.

        Returns
        -------
        list of tuple of str
        """
        run = NOT_RECORDED if self.run is None else self.run
        if self.tone:
            run += f', tone {self.tone}'
            if self.onset is not None and self.run is not None:
                run += f' found at {self.onset:.4g} s'
        sweep = NOT_RECORDED if not self.sweep else '; then '.join(self.sweep)
        if self.drift_hz:
            # the clock the levels were read on was the specification's,
            # corrected toward the run's; said, since a sweep that
            # drifted is not the sweep specified
            sweep += (f'; sweep clock corrected by {self.drift_hz:+.3g} Hz '
                      'at its end')
        detector = (NOT_RECORDED if self.cycles is None else
                    f'Vold-Kalman filter tracking each tone, {self.cycles:g} '
                    "cycles of the tone's frequency per reading")
        return list(zip(self.ROWS, (
            _names(self.channels), _places(self.channels), run, sweep,
            detector, self._distortion()), strict=True))

    def text(self) -> str:
        """The caption as one paragraph, 'Label: text' rows joined by
        '. '.

        Returns
        -------
        str
        """
        return _paragraph(self.rows())

    def _distortion(self) -> str:
        if self.distortion is None:
            return 'not checked'
        if isinstance(self.distortion, str):
            return f'not judged: {self.distortion}'
        said = []
        for channel in self.channels:
            if channel.dof in self.distortion.dofs:
                # the check's own statement of each channel's verdict
                said.append(self.distortion.describe(channel.dof))
            else:
                said.append(f'{channel.dof}: not read by the check')
        return '; '.join(said)


def sweep_segments(tone: Any) -> tuple[str, ...]:
    """Each segment of a specified tone in words: 'logarithmic, 1 oct/min,
    ascending 5 Hz to 500 Hz', or 'dwell at 40 Hz for 30 s'.

    Parameters
    ----------
    tone : SineTone
        The tone as specified.

    Returns
    -------
    tuple of str
    """
    from .sine import LINEAR

    seconds = tone.segment_seconds()
    out = []
    for k, rate in enumerate(tone.segment_rate):
        f0, f1 = float(tone.frequency[k]), float(tone.frequency[k + 1])
        if tone.is_dwell(k):
            out.append(f'dwell at {f0:.4g} Hz for {float(seconds[k]):.4g} s')
            continue
        law, unit = (('linear', 'Hz/s') if tone.segment_type[k] == LINEAR
                     else ('logarithmic', 'oct/min'))
        direction = 'ascending' if f1 > f0 else 'descending'
        out.append(f'{law}, {abs(float(rate)):.4g} {unit}, {direction} '
                   f'{f0:.4g} Hz to {f1:.4g} Hz')
    return tuple(out)


def sine_caption(levels: Any, records: int | Sequence[int] = 0, *,
                 tone: str | None = None, run: str | None = None,
                 channel_table: Any = None, geometry: Any = None,
                 specification: Any = None,
                 distortion: Any = None) -> SineCaption:
    """The caption a report plot of a sine sweep's extracted level
    carries, read off the records drawn and the objects around them.

        caption = sine_caption(levels, 0, tone='Up', run='Run 2',
                               specification=spec)
        caption.text()

    The sweep is the tone as the requirement specifies it, segment by
    segment, with the correction the extraction made to its clock; the
    detector is the Vold-Kalman filter the extraction tracks each tone
    with, and the smoothing a set of levels was read with
    (`SineLevelSet.cycles`). The distortion is a check already made
    (`sine_distortion.check_distortion`), passed in: it rebuilds every
    tone on every channel, which is too slow to make for a caption, so
    a caption with none says it was not checked. Which sweep of a run's
    several this was is not held by a level, so the caption does not
    say.

    Parameters
    ----------
    levels : SineLevel or SineLevelSet
        The extracted level, or the set holding it.
    records : int or sequence of int, default 0
        The records (control channels) the plot draws.
    tone : str, optional
        For a set, the tone drawn; its first when omitted.
    run : str, optional
        The time history the level was extracted from.
    channel_table : ChannelTable, optional
        The channel number and comment.
    geometry : Geometry, optional
        The global axis each measured direction is nearest.
    specification : SineSweepSpecification, optional
        The requirement drawn, which specifies the sweep.
    distortion : DistortionCheck or str, optional
        The distortion check of the tone, or why it could not be judged.

    Returns
    -------
    SineCaption

    Raises
    ------
    TypeError
        If `levels` is not a sine level or a set of them.
    KeyError
        If the set holds no tone `tone`.
    IndexError
        If a record is not one of the level's, or none is asked for.
    """
    from .sine import SineLevelSet

    if caption_kind(levels) != 'sine':
        raise TypeError('a sine caption reads an extracted sine level; '
                        f'{type(levels).__name__} is not one')
    cycles = None
    if isinstance(levels, SineLevelSet):
        cycles = levels.cycles
        if not levels.levels:
            raise IndexError('the set holds no tone to caption')
        level = levels.tone(tone) if tone else levels.levels[0]
    else:
        level = levels
    wanted = _records_wanted(level, records)
    sweep = None
    if specification is not None:
        try:
            sweep = sweep_segments(specification.tone(level.tone))
        except KeyError:
            sweep = None        # a tone the requirement does not name
    return SineCaption(
        channels=_channels(level, wanted, channel_table, geometry),
        tone=level.tone, run=run, onset=float(level.onset), sweep=sweep,
        drift_hz=float(level.drift_hz), cycles=cycles,
        distortion=distortion)


@dataclass(frozen=True)
class TimeCaption:
    """The facts a report plot of a time history carries, read off its
    records by `time_caption`.

    Attributes
    ----------
    channels : tuple of CaptionChannel
        The channels drawn.
    run : str or None
        The run: the time history's name.
    span : tuple of float or None
        The first and last instant recorded, in seconds.
    stretch : tuple of float or None
        The (start, stop) of the stretch analyzed (the record's
        averaging), in seconds; None when no averaging is set.
    sample_rate : float or None
        The sample rate in Hz; None for fewer than two samples or a
        record not evenly sampled.
    ranges : tuple of str or None
        Each channel's measurement range, in words: its full scale from
        the channel table, or why it has none; None without a table.
    """

    #: the rows of a time-history caption, in the order they are read
    ROWS: ClassVar[tuple[str, ...]] = (
        'Channel', 'Location and axis', 'Run', 'Sample rate', 'Range')
    #: the rows that belong to one channel
    CHANNEL_ROWS: ClassVar[tuple[str, ...]] = (
        'Channel', 'Location and axis', 'Range')

    channels: tuple[CaptionChannel, ...]
    run: str | None = None
    span: tuple[float, float] | None = None
    stretch: tuple[float, float] | None = None
    sample_rate: float | None = None
    ranges: tuple[str, ...] | None = None

    def rows(self) -> list[tuple[str, str]]:
        """The caption as (label, text) rows, one per `ROWS` entry.

        Returns
        -------
        list of tuple of str
        """
        run = NOT_RECORDED if self.run is None else self.run
        if self.span is not None:
            run += f', {self.span[0]:.4g} s to {self.span[1]:.4g} s recorded'
        if self.stretch is not None:
            run += (f', {self.stretch[0]:.4g} s to {self.stretch[1]:.4g} s '
                    'analyzed')
        rate = (NOT_RECORDED if self.sample_rate is None
                else f'{self.sample_rate:.6g} Hz')
        ranges = (NOT_RECORDED if self.ranges is None
                  else '; '.join(self.ranges))
        return list(zip(self.ROWS, (
            _names(self.channels), _places(self.channels), run, rate,
            ranges), strict=True))

    def text(self) -> str:
        """The caption as one paragraph, 'Label: text' rows joined by
        '. '.

        Returns
        -------
        str
        """
        return _paragraph(self.rows())


def time_caption(history: Any, records: int | Sequence[int] = 0, *,
                 run: str | None = None, channel_table: Any = None,
                 geometry: Any = None, unit_system: Any = None) -> TimeCaption:
    """The caption a report plot of a time history carries, read off the
    records drawn and the objects around them.

        caption = time_caption(history, [0, 1], run='Run 4',
                               channel_table=table)
        caption.text()

    The span and the sample rate are the record's own, and the stretch
    analyzed is its averaging. The range of each channel is its full
    scale from the channel table, the level it clips at in its own
    units (`overload.table_full_scales`), so a reader can set a trace's
    peaks against it.

    Parameters
    ----------
    history : TimeHistory
        The measured record; a transient specification is refused.
    records : int or sequence of int, default 0
        The records the plot draws.
    run : str, optional
        The run's name.
    channel_table : ChannelTable, optional
        The channel number, comment and range.
    geometry : Geometry, optional
        The global axis each measured direction is nearest.
    unit_system : UnitSystem, optional
        The display units the ranges are written in.

    Returns
    -------
    TimeCaption

    Raises
    ------
    TypeError
        If `history` is not a measured `TimeHistory`.
    IndexError
        If a record is not one of the history's, or none is asked for.
    """
    from ..units import DEFAULT_SYSTEM

    if caption_kind(history) != 'time':
        raise TypeError('a time-history caption reads a measured time '
                        f'history; {type(history).__name__} is not one')
    wanted = _records_wanted(history, records)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    t = np.asarray(history.abscissa, dtype=float)
    span = (float(t[0]), float(t[-1])) if t.size else None
    rate = stretch = None
    if t.size > 1:
        try:
            rate = float(history.sample_rate)
        except ValueError:
            rate = None         # not evenly sampled: no one rate to state
    averaging = getattr(history, 'averaging', None)
    if averaging is not None and rate:
        stretch = (span[0] + averaging.start_sample(rate) / rate,
                   averaging.stop(rate, span[0]))
    ranges = None
    if channel_table is not None:
        from .overload import table_full_scales

        scales = table_full_scales(channel_table, history)
        said = []
        for record in wanted:
            value = float(scales.full_scales[record])
            dimension = str(history.ordinate_dim[record])
            name = str(history.response_dof[record])
            if not np.isfinite(value):
                said.append(f'{name}: {scales.reasons[record]}')
                continue
            unit = us.label_text(dimension)
            said.append(f'{name}: ±{float(us.from_si(value, dimension)):.4g}'
                        f'{" " + unit if unit else ""} full scale')
        ranges = tuple(said)
    return TimeCaption(
        channels=_channels(history, wanted, channel_table, geometry),
        run=run, span=span, stretch=stretch, sample_rate=rate,
        ranges=ranges)
