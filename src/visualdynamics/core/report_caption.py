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
  spectrum was averaged; a banded spectrum carries the line spacing and
  the degrees of freedom of the narrowband spectrum it was banded
  from);
- the window, the overlap, the frames and the stretch of the record
  are the `Averaging` the spectrum was made with, which a spectrum does
  not carry itself (`Project.averaging_of` gives it from the
  provenance);
- the requirement is the specification drawn with the spectrum: which
  limits it has, and the overall rms the record measured beside the one
  the specification asks for, over the band it is written.

A fact nothing records is written as "not recorded" rather than left
out, so a reader sees the gap instead of assuming the line was never
needed. `Project.caption_of` gathers the record's surroundings from a
project and returns the caption; the report draws it under a plot
block that asks for it, and `plot.plot_captioned` under a saved plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
    """

    dof: str
    quantity: str
    channel: int | None = None
    node: int | None = None
    direction: str = ''
    description: str = ''
    nearest_axis: str | None = None
    axis_angle: float | None = None

    def name(self) -> str:
        """'channel 3, 110Z+ acceleration', or the DOF and quantity
        alone when no channel table numbers it."""
        what = f'{self.dof} {self.quantity}'.strip()
        return what if self.channel is None else f'channel {self.channel}, {what}'

    def place(self) -> str:
        """The location and the axis: 'node 110 (bracket), Z+ (nearest
        global Z+)'."""
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
    pooled : int or None
        Records of the same channel pooled into each average, as a
        capture saved one record per average is.
    dof : tuple of float or None
        The equivalent statistical degrees of freedom of each channel's
        average, discounted for overlap.
    nominal_dof : tuple of int or None
        Two per frame averaged, the count without the discount.
    requirement : str or None
        The name of the specification drawn, None when none is.
    limits : tuple of str
        Which of the specification's limit pairs are drawn: 'warning',
        'abort', or both.
    measured_rms, reference_rms : float or None
        The overall rms of the one record captioned, scaled as it is
        drawn, and the rms the specification asks for, both over the
        band the comparison judged (`compliance.compare`), in
        `rms_unit`.
    rms_unit : str
        The display unit of the two rms values.
    rms_band : tuple of float or None
        The (low, high) band in Hz the two rms values are taken over.
    scale_db : int
        The decibels the measured curve is drawn scaled by against the
        specification (`compliance.comparison_scale_db`); 0 for none.
    """

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
    pooled: int | None = None
    dof: tuple[float, ...] | None = None
    nominal_dof: tuple[int, ...] | None = None
    requirement: str | None = None
    limits: tuple[str, ...] = ()
    measured_rms: float | None = None
    reference_rms: float | None = None
    rms_unit: str = ''
    rms_band: tuple[float, float] | None = None
    scale_db: int = 0

    def rows(self) -> list[tuple[str, str]]:
        """The caption as (label, text) rows, one per `ROWS` entry, in
        that order.

        Returns
        -------
        list of tuple of str
            Each row's label and what it says.
        """
        return list(zip(ROWS, (self._channel(), self._place(), self._run(),
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
            if self.pooled and self.pooled > 1:
                said += f' in each of {self.pooled} records'
        if self.per_octave is not None:
            said += f', banded to 1/{self.per_octave} octave'
        return said

    def _resolution(self) -> str:
        lines = (None if self.line_spacing is None
                 else f'{self.line_spacing:.4g} Hz lines')
        if self.per_octave is not None:
            bands = f'1/{self.per_octave}-octave bands'
            return bands if lines is None else f'{bands}, from {lines}'
        return lines or 'lines not evenly spaced'

    def _window(self) -> str:
        if self.window is None:
            return NOT_RECORDED
        said = self.window
        if self.overlap is not None:
            said += f', {self.overlap * 100:.4g}% overlap'
        if self.frame_length is not None:
            said += f', frames of {self.frame_length} samples'
        return said

    def _dof(self) -> str:
        if not self.dof:
            return NOT_RECORDED
        said = _span(self.dof) + ' equivalent'
        if self.nominal_dof:
            said += f' ({_span(self.nominal_dof)} nominal)'
        if self.per_octave is not None:
            # banding sums lines and raises the count, which is why a
            # banded spectrum carries none of its own; what is known is
            # the count of each line that went into a band
            said += ' per narrowband line, before banding'
        return said

    def _requirement(self) -> str:
        if self.requirement is None:
            return 'none drawn'
        drawn = ' and '.join(self.limits)
        said = (f'{self.requirement}, reference with its {drawn} limits drawn'
                if drawn else f'{self.requirement}, reference drawn')
        if self.measured_rms is not None and self.reference_rms is not None:
            unit = f' {self.rms_unit}' if self.rms_unit else ''
            said += (f'; {self.measured_rms:.3g}{unit} rms measured against '
                     f'{self.reference_rms:.3g}{unit} rms asked')
            if self.rms_band is not None:
                said += (f', {self.rms_band[0]:.4g} Hz to '
                         f'{self.rms_band[1]:.4g} Hz')
        if self.scale_db:
            said += (f'; measured scaled {self.scale_db:+d} dB for the '
                     'comparison')
        return said


def _span(values: Sequence[float]) -> str:
    """One number when every channel has the same, else 'low to high'."""
    low, high = float(min(values)), float(max(values))
    if np.isclose(low, high, rtol=1e-9, atol=0.0):
        return f'{low:.4g}'
    return f'{low:.4g} to {high:.4g}'


def _spacing(abscissa: np.ndarray) -> float | None:
    """The line spacing of an evenly spaced axis, else None."""
    steps = np.diff(np.asarray(abscissa, dtype=np.float64))
    if steps.size == 0 or not np.all(np.isfinite(steps)) or steps[0] <= 0.0:
        return None
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        return None
    return float(steps[0])


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
    from .data import channel_quantities, parse_dof

    dof = str(spectrum.response_dof[record])
    dimension = str(spectrum.known_dim(record))
    quantity = '' if dimension == 'unknown' else channel_quantities(dimension)[0]
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
    autos = [spectrum.record_pair(record)[0] == spectrum.record_pair(record)[1]
             for record in wanted]
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
        pooled = counts[0]

    if run is None and spectrum.block is not None:
        labels = {spectrum.block[record] for record in wanted}
        run = ', '.join(sorted(labels)) or None
    stretch = None
    if averaging is not None and sample_rate:
        stretch = (origin + averaging.start_sample(sample_rate) / sample_rate,
                   averaging.stop(sample_rate, origin))

    limits: tuple[str, ...] = ()
    measured_rms = reference_rms = None
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
                if compared.get('lines'):
                    band = tuple(float(edge) for edge in compared['band'])
                    dimension = str(spectrum.known_dim(mi))
                    measured_rms = float(compared['measured_rms'])
                    reference_rms = float(compared['specification_rms'])
                    if dimension != 'unknown':
                        amplitude = channel_quantities(dimension)[0]
                        measured_rms = float(us.from_si(measured_rms, amplitude))
                        reference_rms = float(us.from_si(reference_rms,
                                                         amplitude))
                        unit = us.label_text(amplitude)
    return RecordCaption(
        channels=channels, kind=kind, run=run, stretch=stretch,
        line_spacing=spacing, per_octave=per_octave,
        window=None if averaging is None else averaging.window,
        overlap=None if averaging is None else averaging.overlap,
        frame_length=None if averaging is None else averaging.frame_length,
        frames=None if averaging is None else averaging.frames,
        pooled=pooled, dof=dof, nominal_dof=nominal,
        requirement=(None if specification is None
                     else requirement or 'specification'),
        limits=limits, measured_rms=measured_rms,
        reference_rms=reference_rms, rms_unit=unit, rms_band=band,
        scale_db=int(scale_db or 0) if specification is not None else 0)
