"""Importer for Rattlesnake vibration controller output files (.nc4).

Rattlesnake streams results to netCDF4 with a channel table that includes
engineering units (read in whatever case they were typed), so imports
are fully unit-aware — no unit declaration
needed. Layout (file_version 3.x):

- root attrs: sample_rate, hardware, file_version, ...
- time_data (response_channels, time_samples), in channel engineering units
- channels group: node_number, node_direction, unit, channel_type, and other
  per-channel metadata
- one group per environment (e.g. 'Random') that may carry a specification
  CPSD matrix (specification_frequency_lines, specification_cpsd_matrix_*)
  and the bands around it (specification_warning_matrix,
  specification_abort_matrix, each (2, lines, channels), lower first)

A run saves twice and the two files share nothing: streaming writes the time
histories, and a separate save writes what the environment computed from
them. Both are read here.

Returns a dict: 'channel_table' -> ChannelTable, and whichever of these the
file holds — 'time_data' -> TimeHistory, '<env>_specification' ->
Specification (diagonal ASDs with their warning and abort limits;
full_cpsd=True for the whole matrix as cross-PSD records), '<env>_frf' ->
Frf, '<env>_coherence' -> MultipleCoherence, '<env>_response_cpsd'
and
'<env>_drive_cpsd' -> Psd.

A modal environment counts every enabled channel as a response and never
excludes its references, so the drives appear among them and the saved
matrix carries each drive against the drives. Those rows are identity and
cross-talk by construction — |H| exactly 1 against itself, numerically zero
against the other, coherence exactly 1 — so they are dropped: they are
arithmetic rather than measurement, and would put a force-per-force axis on
the plot beside the real one.

A spectral file carries neither time data nor, once rattlesnake's own random
save has been through it, the root attributes — so the sample rate may be
missing and the frequency axis has to come from the specification's own
lines. The noise cross spectra a random save also writes are deliberately
left alone: they describe the measurement's noise floor and would double the
object count for something rarely looked at.

Reading only, and deliberately: a `.nc4` records a controller run — hardware
settings, environments, the lot — that visualdynamics does not hold, so a file written
from here would describe a test that never happened. See docs/export.md.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from ..core.channel_table import ChannelTable
from ..core.data import (
    Frf,
    MultipleCoherence,
    Psd,
    Specification,
    TimeHistory,
    TransientSpecification,
)
from ..core.sine import SineSweepSpecification, SineTone
from ..units import UNKNOWN, dimension_of, fold_unit_case, si_factor


def sniff(path: str | os.PathLike) -> bool:
    if not str(path).lower().endswith(('.nc4', '.nc')):
        return False
    try:
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            if 'channels' not in ds.groups:
                # a saved system-ID package: no channel table, one
                # group per environment holding the measured plant
                return any('frf_data_real' in group.variables
                           for group in ds.groups.values())
            # a run saves two files: the streamed time data, and what the
            # environment computed from it. Neither has the other's contents.
            if 'time_data' in ds.variables:
                return True
            return any(name in group.variables
                       for group in ds.groups.values()
                       for name in _SPECTRAL_MARKERS)
    except Exception:  # noqa: BLE001 - sniffers must not raise on foreign files
        return False


# what a saved spectrum is made of, in either environment's spelling
_SPECTRAL_MARKERS = ('frf_data_real', 'coherence', 'frf_coherence')


#: Rattlesnake's own EnvironmentType, as the number it writes into the
#: file's `environment_types` variable. This is what makes a saved run
#: self-describing: an environment's *group* is named by whoever set the
#: test up, so a group called 'Random' is only a random environment by
#: convention, and often is not one at all.
ENVIRONMENT_KINDS = {1: 'random', 2: 'transient', 3: 'sine', 4: 'time',
                     6: 'modal'}

#: an environment that records without driving anything. A time-history
#: replay running beside a shaker test says nothing about what kind of
#: test it is, so it does not on its own make a run mixed-mode.
PASSIVE_KINDS = frozenset({'time'})

#: the visualdynamics project each kind of run is.
# A transient run replicates a *waveform*, which is what the controller
# can do today; a shock test meets an *SRS*, which it cannot. The two
# were one project type here, and calling a transient run a shock test is
# exactly what makes the difference hard to see.
PROJECT_FOR_KIND = {'modal': 'Modal Test', 'random': 'Random Vibration',
                    'transient': 'Transient', 'sine': 'Sine Sweep'}

#: which half of a mixed run names the project, in order. Both halves'
#: specifications import and both analyses work whatever the type says;
#: the type only decides which slots and report the tree leads with,
#: and the person can switch. Random leads because its workflow is the
#: established one; sine follows.
MIXED_PRECEDENCE = ('random', 'sine', 'transient', 'modal')

#: how far under a record's own top level a detected stretch may sit and
#: still be taken for the test, in dB. A run's levels step by 3 and 6;
#: the noise floor under one is tens.
FULL_LEVEL_MARGIN = 9.0


def _kind_of(value):
    """The kind one entry of `environment_types` names, or None.

    Rattlesnake writes the code, so the code is what is read. A file
    naming the kind outright is still answered rather than crashing the
    whole import on `int('transient')` — the names are the ones visualdynamics
    already uses, so there is nothing to guess at, and an entry that is
    neither is no environment visualdynamics knows.
    """
    try:
        return ENVIRONMENT_KINDS.get(int(value))
    except (TypeError, ValueError):
        name = str(value).strip().lower()
        return name if name in set(ENVIRONMENT_KINDS.values()) else None


def _environment_kinds(ds):
    """{environment name: kind} an open dataset says it holds."""
    types = ds.variables.get('environment_types')
    names = ds.variables.get('environment_names')
    if types is None:
        return {}
    found = [_kind_of(value) for value in types[:]]
    labels = (_strings(names) if names is not None
              else [f'Environment {i}' for i in range(len(found))])
    return {label: kind for label, kind in zip(labels, found)
            if kind is not None}


def environment_kinds(path: str | os.PathLike) -> dict[str, str]:
    """{environment name: kind} the file says it holds.

    Empty for a file that does not say — an nc4 assembled by hand, or an
    older save from before the controller wrote its types down.
    """
    import netCDF4

    with netCDF4.Dataset(str(path)) as ds:
        return _environment_kinds(ds)


def _run_kind(kinds: Iterable[str]) -> str | None:
    """What one set of environment kinds amounts to."""
    kinds = set(kinds)
    if not kinds:
        return None
    driving = kinds - PASSIVE_KINDS
    if not driving:
        return 'time'
    if len(driving) > 1:
        return 'mixed'
    return driving.pop()


def run_kind(path: str | os.PathLike) -> str | None:
    """What kind of test the file holds, by its own account.

    'modal', 'random', 'transient', 'sine' or 'time' for a run of one
    kind; 'mixed' when more than one environment drove the article at
    once, which is what a random-plus-sine-sweep run is; None when the
    file does not say.
    """
    return _run_kind(environment_kinds(path).values())


def project_type(path: str | os.PathLike) -> str | None:
    """The visualdynamics project this file is a run of, or None if visualdynamics has no
    project of that kind — or the file never said what kind it was.

    A mixed run answers with its leading half (`MIXED_PRECEDENCE`):
    every environment's specification imports regardless, so the type
    only chooses which workflow the tree leads with.
    """
    import netCDF4

    with netCDF4.Dataset(str(path)) as ds:
        kinds = set(_environment_kinds(ds).values())
        # a saved system-ID package carries no environment_types — it
        # is the measured plant alone, and that is its own kind of
        # test. A *streamed* sysid save is indistinguishable from a
        # run of its environment and keeps that type; the person can
        # switch it to System ID.
        if not kinds and any('frf_data_real' in group.variables
                             for group in ds.groups.values()):
            return 'System ID'
    kind = _run_kind(kinds)
    if kind == 'mixed':
        kind = next((k for k in MIXED_PRECEDENCE if k in kinds), None)
    return PROJECT_FOR_KIND.get(kind)


def streamed_sysid_candidate(path: str | os.PathLike) -> bool:
    """Whether a streamed save has the shape a system ID leaves:
    exactly two streams, a quiet one then a loud one — the ambient
    measurement and the driven excitation.

    The file itself cannot settle the question (a run of the
    environment that stopped and restarted its stream once looks the
    same on paper), so this is grounds to *ask the person*, never to
    decide. The quiet-then-loud check is what keeps the question from
    being asked about every two-stream file: a restarted run's halves
    play at one level, a system ID's differ by the whole test. Ten
    decibels is well under any real ambient-to-driven gap and well
    over a level change within one run.
    """
    if not str(path).lower().endswith(('.nc4', '.nc')):
        return False
    try:
        import netCDF4
        import numpy as np

        with netCDF4.Dataset(str(path)) as ds:
            if ('time_data' not in ds.variables
                    or 'time_data_1' not in ds.variables
                    or 'time_data_2' in ds.variables):
                return False
            quiet = float(np.asarray(ds.variables['time_data'][()]).std())
            driven = float(
                np.asarray(ds.variables['time_data_1'][()]).std())
        # a simulated ambient can be exact silence; that is the
        # extreme of the same shape, not a different case
        return driven > 10 ** (10 / 20) * quiet and driven > 0.0
    except Exception:  # noqa: BLE001 - sniffers must not raise on foreign files
        return False


def _strings(var):
    return [str(v) for v in var[:]]


# what an environment calls its averaging, in the order we would rather
# have it. A random environment describes the control loop first and its
# system identification second; both are real settings the run used, and
# a file that has only the sysid ones still knows how it was averaged.
AVERAGING_KEYS = (
    # (frame length, frame count, overlap, window)
    ('samples_per_frame', 'num_averages', 'overlap', 'frf_window'),
    ('samples_per_frame', 'frames_in_cpsd', 'cpsd_overlap', 'cpsd_window'),
    ('sysid_frame_size', 'sysid_averages', 'sysid_overlap', 'sysid_window'),
)


def _averaging(ds, samples, sample_rate):
    """How the run itself averaged, if the file says.

    Every environment in the file is asked, and the first complete
    answer is taken. The frame count is trimmed to what the saved record
    can actually carry: a control loop may have averaged hundreds of
    frames over a run whose saved capture holds ten.

    A transient environment is asked first and answered differently.
    Its repeats are not described anywhere in the file — `repeat` is a
    runtime instruction that never reaches the netCDF — but they do not
    need to be: each one plays the control signal, so the frame is that
    signal's length, they run back to back, and the window is
    rectangular. Taking that route matters because the fallback below
    would find `sysid_frame_size` and friends and read them as the
    answer. They are not: they describe the random excitation
    Rattlesnake runs beforehand to measure the FRF it inverts, and as
    averaging parameters for the transient they are wrong three ways —
    Hann for rectangular, half overlap for none, and a frame a quarter
    of the true one.
    """
    from ..core.averaging import Averaging

    for group in ds.groups.values():
        if 'control_signal' not in group.variables:
            continue
        length = int(group.variables['control_signal'].shape[-1])
        if 2 <= length <= samples:
            return Averaging(frame_length=length, overlap=0.0,
                             window='rectangle',
                             frames=samples // length)

    # key sets outrank group order: a mixed run's sine environment
    # carries sysid_* attributes (the plant-measurement phase), and
    # with groups outermost those shadowed the random loop's own
    # samples_per_frame/frames_in_cpsd whenever the sine group came
    # first in the file
    for length_key, count_key, overlap_key, window_key in AVERAGING_KEYS:
        for group in ds.groups.values():
            attrs = set(group.ncattrs())
            if not {length_key, count_key}.issubset(attrs):
                continue
            length = int(group.getncattr(length_key))
            if length < 2 or length > samples:
                continue
            averaging = Averaging(
                frame_length=length,
                overlap=float(group.getncattr(overlap_key))
                if overlap_key in attrs else 0.0,
                window=group.getncattr(window_key)
                if window_key in attrs else 'rectangle',
                frames=max(int(group.getncattr(count_key)), 1))
            room = averaging.most_frames(samples, sample_rate)
            if room < 1:
                continue
            return replace(averaging, frames=min(averaging.frames, room))
    return None


def _started(history, averaging, kind):
    """The same averaging, moved to where the record is worth averaging.

    The file settles the frame length, the count, the overlap and the
    window; the one thing it never says is *when*. Left at zero the
    analysis starts on the shaker coming up, which is the one stretch of
    a run that is certainly not the test.

    Only for a random run. A modal survey averages bursts, and a burst
    record has no stationary stretch to find — asked for one on the
    airplane's modal capture the detector returns sixteen seconds
    holding four of the twenty averages the file asked for, which is
    worse than starting at zero.

    What is wanted here is specifically the full-level stretch: a
    controller averages its run at full level and nowhere else. The
    detector answers a slightly wider question — where can these frames
    be had — and on a record holding eight seconds of test inside eighty
    of quiet the quiet is where they can be had. So what it finds is
    checked against the record's own top level, and a stretch well below
    it is one the detector has settled for rather than found.

    Detection runs at the detector's own resolution rather than the
    file's frame length. Finding which side of a step at 61 s a hop
    falls does not want a four-second hop; the file's frames are laid
    down from wherever a second-long one says.

    The move also needs room for the frames the file asked for. Short of
    that the detector has plainly not found what the controller
    averaged, and the file's own account is the better one.
    """
    if kind != 'random':
        return averaging
    from ..core.detect import TOP_QUANTILE, compress, plateau, trim

    try:
        level, per_frame, hop, sigma = compress(history)
    except ValueError:
        # unevenly sampled, or too short to frame at all
        return averaging
    if len(level) == 0:
        return averaging
    rate = history.sample_rate
    # the analysis spans this many samples; the detector counts in hops
    wanted_hops = -(-averaging.span // hop)
    first, last = trim(*plateau(level, sigma, max(wanted_hops, per_frame)))
    if last <= first:
        return averaging
    # is this the test, or the quietest thing long enough to hold the
    # frames? A stretch a long way under the record's top level is the
    # second, and starting the analysis there analyses the noise floor
    top = float(np.percentile(level, TOP_QUANTILE))
    if float(np.median(level[first:last])) < top - FULL_LEVEL_MARGIN:
        return averaging
    moved = replace(averaging, start=first * hop / rate)
    if moved.most_frames(last * hop, rate) < averaging.frames:
        return averaging
    return moved


def _frame_count(ds, samples):
    """How many captures the time data is a stack of, or 0 if it is a stream.

    A run saves twice and the two files hold different things under the same
    variable name. Streaming writes a *recording*: whatever arrived between
    Start Streaming and Stop Streaming, one continuous span. A modal
    environment's spectral save writes a *stack*: its UI appends one frame
    per accepted average, so the samples are separate captures laid end to
    end, and drawing them as one history shows a signal that never existed —
    the excitation restarts at every frame boundary.

    Telling them apart needs both halves of the evidence. The environment
    must say how it framed (`samples_per_frame`, `num_averages`) and the
    sample count must be exactly that many whole frames; and the file must
    carry spectral results, because a stream can be an exact multiple of the
    frame size by luck. Anything less certain is left as a recording.

    Overlap does not enter into it. The save writes whole frames whatever
    the overlap is, so at 50% the repeated samples are already in the file
    and splitting copies nothing that is not there.
    """
    for group in ds.groups.values():
        if not any(name in group.variables for name in _SPECTRAL_MARKERS):
            continue
        per_frame = getattr(group, 'samples_per_frame', 0)
        averages = getattr(group, 'num_averages', 0)
        if per_frame and averages and samples == per_frame * averages:
            return int(averages)
    return 0


def _unit_scale(unit):
    """(SI scale factor, dimension tag, normalized unit) for a channel unit.

    Channels whose unit string visualdynamics cannot interpret import unit-less rather
    than being silently treated as dimensionless. The case is the
    controller's user's, not pint's: a table typed `G` or `LBF` is read
    as g and lbf, and the record carries that spelling
    (`units.fold_unit_case`).
    """
    normalized = fold_unit_case(unit.replace('^', '**').strip())
    dim = dimension_of(normalized) if normalized else None
    if dim is None:
        return 1.0, UNKNOWN, None
    return si_factor(normalized), dim, normalized


def _frequency_lines(group, sample_rate):
    """The frequency axis a spectral result sits on.

    A spectral save records the lines only for a specification; everything
    else is implicitly 0..Nyquist over however many lines it has, which is
    what the environment's frame size makes it.
    """
    if 'specification_frequency_lines' in group.variables:
        return np.asarray(group.variables['specification_frequency_lines'][()])
    lines = len(group.dimensions['fft_lines'])
    if sample_rate is None:
        # nothing in the file says; index the lines rather than invent a rate
        return np.arange(float(lines))
    return np.linspace(0.0, sample_rate / 2, lines)


def _response_transformation(group, indices, scales_dims):
    """The environment's response transformation, if it ran one:
    (matrix, row labels, the rows' (scale, dimension, unit)), or None.

    A random, sine or transient environment may control a *virtual*
    response — a matrix over the control channels, applied by the
    controller's data collector to every frame it collects
    (`response_frame = T @ response_frame`) — and then everything the
    environment computes and saves is over the matrix's rows: the
    specification, the FRF, the coherence, the control CPSD. Only the
    streamed time data stays in hardware channels. The file names the
    rows nowhere (`load_specification` drops the coordinates when a
    transformation is set), so they import as node-only DOFs numbered
    from 1, the same way the system-ID package numbers channels it
    cannot name (Brandon, 2026-09-18: three virtual degrees of freedom
    from twelve accelerometers, labelled 1, 2 and 3).

    Units: the controller multiplies engineering-unit values, so a row
    is in the control channels' unit exactly when they all share one;
    a mix of units gives rows that are numbers in no unit, imported
    raw and undeclared rather than labelled with a guess. (A rotation
    row of a rigid-body transformation is in that unit per length, and
    the file cannot say which rows those are — the user declares.)
    """
    if 'response_transformation_matrix' not in group.variables:
        return None
    matrix = np.asarray(group.variables['response_transformation_matrix'][()],
                        dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(indices):
        raise ValueError(
            f'{group.name}: the response transformation is '
            f'{matrix.shape} but the environment controls '
            f'{len(indices)} channels')
    labels = [str(k + 1) for k in range(matrix.shape[0])]
    kinds = {scales_dims[ci] for ci in indices}
    common = kinds.pop() if len(kinds) == 1 else (1.0, UNKNOWN, None)
    return matrix, labels, common


def _control_channels(group, indices, dofs, scales_dims):
    """What the environment's specification and spectra are over:
    [(dof, (scale, dim, unit)), ...] — the control channels themselves,
    or the rows of the response transformation when there is one."""
    found = _response_transformation(group, indices, scales_dims)
    if found is None:
        return [(dofs[ci], scales_dims[ci]) for ci in indices]
    _matrix, labels, common = found
    return [(label, common) for label in labels]


def _spectra(env_name, group, dofs, scales_dims, sample_rate, drives):
    """FRFs, coherence and cross spectra computed by an environment.

    Rattlesnake's two environments spell these differently — modal writes
    `coherence` against its reference channels, random writes
    `frf_coherence` against its control channels — so the channel indices
    come from whichever list the group carries.
    """
    if 'fft_lines' not in group.dimensions:
        return {}
    out = {}
    freq = _frequency_lines(group, sample_rate)

    def indices(*names: str) -> np.ndarray | None:
        for name in names:
            if name in group.variables:
                return np.asarray(group.variables[name][()], dtype=int)
        return None

    responses = indices('response_channel_indices', 'control_channel_indices')
    references = indices('reference_channel_indices')
    if references is None and 'drive_channels' in group.dimensions:
        # A random environment names its control channels but not its
        # drives, so the drives come from the channel table, which marks
        # them with a feedback device. Inferring them instead as "every
        # channel that is not a control channel" happens to be right only
        # when the file holds nothing but controls and drives — the moment a
        # test instruments more than it controls, it returns hundreds of
        # them and the FRF slicing walks off the end of its drive axis.
        references = np.asarray(drives, dtype=int)
    if references is not None and 'drive_channels' in group.dimensions:
        declared = len(group.dimensions['drive_channels'])
        if len(references) != declared:
            raise ValueError(
                f'{env_name}: found {len(references)} drive channels but the '
                f'spectra were computed against {declared}')

    # Rattlesnake's modal environment takes every enabled channel as a
    # response and never excludes the references, so the drives appear among
    # them. Against a reference set containing itself a channel is identity —
    # |H| exactly 1, coherence exactly 1 — which is arithmetic, not
    # measurement, and would put a spurious force-per-force axis on the plot.
    drives = set(references.tolist()) if references is not None else set()

    # each response row as (dof, (scale, dim, unit), is a drive): the
    # control channels, or the transformation's rows in their place
    rows = None
    if responses is not None:
        found = _response_transformation(group, responses, scales_dims)
        if found is None:
            rows = [(dofs[ri], scales_dims[ri], ri in drives)
                    for ri in responses]
        else:
            rows = [(label, found[2], False) for label in found[1]]

    if 'frf_data_real' in group.variables and rows is not None:
        frf = (np.asarray(group.variables['frf_data_real'][()])
               + 1j * np.asarray(group.variables['frf_data_imag'][()]))
        if frf.shape[1] != len(rows):
            raise ValueError(
                f'{env_name}: the FRF has {frf.shape[1]} response rows '
                f'but the environment names {len(rows)}')
        records, response_dof, reference_dof = [], [], []
        dims, units, ref_units = [], [], []
        for i, (dof_i, (si, di, ui), is_drive) in enumerate(rows):
            if is_drive:
                continue
            for j, rj in enumerate(references):
                sj, dj, uj = scales_dims[rj]
                records.append(frf[:, i, j] * si / sj)
                response_dof.append(dof_i)
                reference_dof.append(dofs[rj])
                known = UNKNOWN not in (di, dj)
                dims.append(f'{di}/{dj}' if known else UNKNOWN)
                units.append(ui if known else None)
                ref_units.append(uj if known else None)
        out[f'{env_name}_frf'] = Frf(
            abscissa=freq, ordinate=np.array(records),
            response_dof=response_dof, reference_dof=reference_dof,
            ordinate_dim=dims, ordinate_unit=units, reference_unit=ref_units)

    for name in ('coherence', 'frf_coherence'):
        if name in group.variables and rows is not None:
            values = np.asarray(group.variables[name][()]).real
            keep = [i for i, (_dof, _sdu, is_drive) in enumerate(rows)
                    if not is_drive]
            # multiple coherence, not ordinary: one curve per response with
            # the references summed over, which is UFF's type 26. Writing
            # it as type 6 promised a reference DOF that is not there.
            out[f'{env_name}_coherence'] = MultipleCoherence(
                abscissa=freq, ordinate=values.T[keep],
                response_dof=[rows[i][0] for i in keep])
            break

    drive_rows = (None if references is None
                  else [(dofs[ci], scales_dims[ci]) for ci in references])
    response_rows = (None if rows is None
                     else [(dof, sdu) for dof, sdu, _drive in rows])
    for prefix, channels, label in (
            ('response_cpsd', response_rows, 'response_cpsd'),
            ('drive_cpsd', drive_rows, 'drive_cpsd')):
        if f'{prefix}_real' not in group.variables or channels is None:
            continue
        matrix = (np.asarray(group.variables[f'{prefix}_real'][()])
                  + 1j * np.asarray(group.variables[f'{prefix}_imag'][()]))
        # the whole matrix, not its diagonal. The cross terms are how two
        # channels move together, which is most of what a CPSD is for; a
        # grid makes the full n x n navigable in a way a flat list did not.
        records, response_dof, reference_dof = [], [], []
        dims, units, ref_units = [], [], []
        for i, (dof_i, (si, di, ui)) in enumerate(channels):
            for j, (dof_j, (sj, dj, uj)) in enumerate(channels):
                records.append(matrix[:, i, j] * si * sj)
                response_dof.append(dof_i)
                reference_dof.append(dof_j)
                known = UNKNOWN not in (di, dj)
                dims.append(
                    (f'{di}**2/frequency' if di == dj
                     else f'{di}*{dj}/frequency') if known else UNKNOWN)
                units.append(ui if known else None)
                ref_units.append(uj if known and ui != uj else None)
        out[f'{env_name}_{label}'] = Psd(
            abscissa=freq, ordinate=np.array(records),
            response_dof=response_dof, reference_dof=reference_dof,
            ordinate_dim=dims, ordinate_unit=units, reference_unit=ref_units)
    return out


def _load_sysid_package(ds, path):
    """A saved system-identification package: the measured plant.

    Save System ID writes the sysid data alone — no channel table, no
    time data — as one group per environment holding the FRF matrix,
    the multiple coherence, and the driven and ambient CPSDs
    ((lines, responses, ...) throughout, layout read from
    `save_package_to_netcdf`). The package records the *hardware
    channel numbers* of its control channels but no node names, so
    responses import as node-only DOFs carrying those numbers ('4'
    for hardware channel 4) — real, if incomplete, and the channel
    table of the run the package came from says the rest. Which
    hardware channels the drives were is not recorded anywhere in the
    file, so drives take 9000-series numbers: visibly synthetic, each
    record's comment saying which drive it is. No units are recorded
    either, so everything imports raw, the FRF's quantities
    undeclared.

    Beyond the FRF and the coherence, the package is the one place
    the *ambient* measurement survives: system ID measures the quiet
    before it drives. The ratio of driven to ambient autospectra
    imports as the driven and ambient PSD pair — divided, where the
    ratio is large the
    FRF is believed; where it approaches one, the measurement is the
    room.
    """
    from ..core.data import Frf, MultipleCoherence, Psd

    out = {}
    for env_name, group in ds.groups.items():
        if 'frf_data_real' not in group.variables:
            continue
        frf = (np.asarray(group.variables['frf_data_real'][()])
               + 1j * np.asarray(group.variables['frf_data_imag'][()]))
        lines, n_responses, n_references = frf.shape
        # the fft line spacing comes from the sysid settings the
        # metadata carries; the specification's own lines are the
        # fallback for a package written without them
        attrs = set(group.ncattrs())
        if {'sysid_sample_rate', 'sysid_frame_size'} <= attrs:
            df = (float(group.getncattr('sysid_sample_rate'))
                  / float(group.getncattr('sysid_frame_size')))
            freq = np.arange(lines) * df
        elif 'specification_frequency_lines' in group.variables:
            freq = np.asarray(
                group.variables['specification_frequency_lines'][()])
        else:
            raise ValueError(
                f'{path}: {env_name} holds a system-ID package but '
                'neither its sysid settings nor specification lines — '
                'the frequency axis cannot be recovered')
        transformed = ('response_transformation_matrix' in group.variables
                       and group.variables['response_transformation_matrix']
                       .shape[0] == n_responses)
        if 'control_channel_indices' in group.variables and not transformed:
            indices = np.asarray(
                group.variables['control_channel_indices'][()], dtype=int)
            responses = [str(int(i) + 1) for i in indices]
        else:
            # no control channels named, or a response transformation
            # whose rows the package is over (`_response_transformation`)
            responses = [str(i + 1) for i in range(n_responses)]
        references = [str(9001 + j) for j in range(n_references)]

        rows, response_dof, reference_dof = [], [], []
        for i in range(n_responses):
            for j in range(n_references):
                rows.append(frf[:, i, j])
                response_dof.append(responses[i])
                reference_dof.append(references[j])
        comments = [f'channel {responses[i]} over drive {j + 1}'
                    for i in range(n_responses)
                    for j in range(n_references)]
        out[f'{env_name}_frf'] = Frf(
            abscissa=freq, ordinate=np.array(rows),
            response_dof=response_dof, reference_dof=reference_dof,
            comment=comments)
        if 'frf_coherence' in group.variables:
            coherence = np.asarray(group.variables['frf_coherence'][()])
            out[f'{env_name}_coherence'] = MultipleCoherence(
                abscissa=freq, ordinate=coherence.T,
                response_dof=list(responses))

        # the driven and ambient autospectra, as the two PSD objects
        # the workflow reads together — the ratio is a *reading* of
        # the pair, not a third object (Brandon, 2026-08-25). The
        # diagonals only: the package's cross terms belong to the
        # controller's estimator, and the ratio never looks at them.
        for which, suffix in (('cpsd', 'excitation_psds'),
                              ('noise_cpsd', 'noise_psds')):
            rows, dofs = [], []
            for prefix, labels in (('response', responses),
                                   ('reference', references)):
                name = f'{prefix}_{which}_real'
                if name not in group.variables:
                    continue
                diag = np.asarray(group.variables[name][()])
                for k, label in enumerate(labels):
                    rows.append(diag[:, k, k])
                    dofs.append(label)
            if rows:
                out[f'{env_name}_{suffix}'] = Psd(
                    abscissa=freq, ordinate=np.array(rows),
                    response_dof=dofs)
    if not out:
        raise ValueError(f'{path}: no system-ID package found')
    return out


def load(path: str | os.PathLike, full_cpsd: bool = False) -> dict[str, Any]:
    import netCDF4

    out = {}
    with netCDF4.Dataset(path) as ds:
        if 'channels' not in ds.groups:
            return _load_sysid_package(ds, path)
        ch = ds.groups['channels']
        nodes = _strings(ch.variables['node_number'])
        directions = _strings(ch.variables['node_direction'])
        units = _strings(ch.variables['unit'])
        num_channels = len(nodes)

        columns = {'channel': np.arange(1, num_channels + 1),
                   'node': nodes, 'direction': directions, 'unit': units}
        for name, var in ch.variables.items():
            if name not in ('node_number', 'node_direction', 'unit'):
                columns[name] = _strings(var)
        # the file states each channel's purpose in its own vocabulary:
        # a channel with a feedback device is a drive — rattlesnake's
        # own rule — and every other enabled channel is measured as a
        # response. Mapping that to a role is translation, not guessing.
        # The file's minimum_value/maximum_value are deliberately NOT
        # mapped into the schema's 'range': they are the acquisition
        # range the DAQ was set to, not the sensor's voltage limit, and
        # the two agreeing is a coincidence of configuration.
        columns['role'] = ['reference' if str(value).strip() else 'response'
                           for value in columns.get(
                               'feedback_device', [''] * num_channels)]
        table = ChannelTable(columns)
        out['channel_table'] = table

        dofs = table.dof_strings()
        scales_dims = [_unit_scale(u) for u in units]
        # a channel is a drive exactly when it has a feedback device, which
        # is how rattlesnake itself decides (`Channel.is_output_channel`)
        feedback = columns.get('feedback_device', [''] * num_channels)
        drive_channels = [i for i, value in enumerate(feedback)
                          if str(value).strip()]

        # a run may hold several streams: stopping and restarting the
        # stream mid-run appends time_data_1, time_data_2... (the
        # streaming process's CREATE_NEW_STREAM), and a system ID run
        # saved with a stream file does exactly that — the ambient
        # noise measurement is one stream and the driven excitation
        # the next. Each imports as its own history; only the first
        # used to, and the driven half of a system ID recording was
        # silently dropped.
        # A spectral file carries a time_samples dimension of zero,
        # which the metadata writer puts there; an empty history is
        # not a history.
        streams = [('time_data', 'time_samples')]
        streams += sorted(
            ((name, f'time_samples_{name.rsplit("_", 1)[1]}')
             for name in ds.variables
             if name.startswith('time_data_')
             and name.rsplit('_', 1)[1].isdigit()),
            key=lambda pair: int(pair[0].rsplit('_', 1)[1]))
        for variable, dimension in streams:
            if (variable not in ds.variables
                    or len(ds.dimensions.get(dimension, ())) == 0):
                continue
            time_data = np.asarray(ds.variables[variable][()])
            sample_rate = float(ds.sample_rate)
            scales = np.array([s for s, _, _ in scales_dims])
            values = time_data * scales[:, np.newaxis]
            frames = _frame_count(ds, time_data.shape[1])
            if frames:
                # (channels, frames * samples) -> one record per capture.
                # reshape only, so the samples are not copied.
                samples = time_data.shape[1] // frames
                values = values.reshape(-1, frames, samples).reshape(-1, samples)
                block = [f'avg {i + 1}' for _dof in dofs
                         for i in range(frames)]
                response_dof = [dof for dof in dofs for _ in range(frames)]
                repeat = frames
            else:
                block, response_dof, repeat = None, dofs, 1
            history = TimeHistory(
                abscissa=np.arange(values.shape[1]) / sample_rate,
                ordinate=values,
                response_dof=response_dof, block=block,
                ordinate_dim=[d for _, d, _ in scales_dims for _ in range(repeat)],
                ordinate_unit=[u for _, _, u in scales_dims for _ in range(repeat)],
            )
            kind = _run_kind(_environment_kinds(ds).values())
            # a pure sine run has no averaging to describe: the sweep
            # is read sample by sample through a tracking filter, and
            # the only frame-shaped attributes in its file are the
            # sysid_* ones — the plant-measurement phase, not the
            # sweep. Shading its frames on the recording claimed an
            # analysis that never happens (Brandon, 2026-08-22).
            averaging = (None if kind == 'sine'
                         else _averaging(ds, values.shape[1],
                                         sample_rate))
            if averaging is not None:
                averaging = _started(history, averaging, kind)
            history.averaging = averaging
            key = ('time_data' if variable == 'time_data'
                   else f'time_data_{int(variable.rsplit("_", 1)[1]) + 1}')
            out[key] = history
            # the virtual responses the environment controlled, as their
            # own history beside the raw channels: the same matrix the
            # controller applied, over the same frames, so the PSDs
            # computed from it are what the specification was judged
            # against (Brandon, 2026-09-18: "a separate time history")
            for env_name, group in ds.groups.items():
                if 'control_channel_indices' not in group.variables:
                    continue
                indices = np.asarray(
                    group.variables['control_channel_indices'][()], dtype=int)
                found = _response_transformation(group, indices, scales_dims)
                if found is None:
                    continue
                matrix, labels, (scale, dim, unit) = found
                virtual = (matrix @ time_data[indices]) * scale
                if frames:
                    virtual = virtual.reshape(-1, frames, samples).reshape(
                        -1, samples)
                    v_block = [f'avg {i + 1}' for _label in labels
                               for i in range(frames)]
                    v_dofs = [label for label in labels for _ in range(frames)]
                else:
                    v_block, v_dofs = None, labels
                transformed = TimeHistory(
                    abscissa=history.abscissa, ordinate=virtual,
                    response_dof=v_dofs, block=v_block,
                    ordinate_dim=[dim] * len(v_dofs),
                    ordinate_unit=[unit] * len(v_dofs),
                    comment=[f'row {label} of the {env_name} response '
                             f'transformation over {len(indices)} control '
                             'channels' for label in v_dofs])
                transformed.averaging = history.averaging
                out[key.replace('time_data', f'{env_name}_transformed')] = \
                    transformed

        # rattlesnake's own spectral save keeps only the environment group;
        # the root attributes, the channel table and the time data all go.
        # So the sample rate may simply not be there.
        rate = (float(ds.sample_rate) if 'sample_rate' in ds.ncattrs()
                else None)
        for env_name, group in ds.groups.items():
            out.update(_spectra(env_name, group, dofs, scales_dims, rate,
                                drive_channels))

        for env_name, group in ds.groups.items():
            if 'specification_frequency_lines' not in group.variables:
                continue
            freq = np.asarray(group.variables['specification_frequency_lines'][()])
            cpsd = (np.asarray(group.variables['specification_cpsd_matrix_real'][()])
                    + 1j * np.asarray(
                        group.variables['specification_cpsd_matrix_imag'][()]))
            indices = np.asarray(group.variables['control_channel_indices'][()],
                                 dtype=int)
            channels = _control_channels(group, indices, dofs, scales_dims)
            if cpsd.shape[-1] != len(channels):
                raise ValueError(
                    f'{env_name}: the specification is over '
                    f'{cpsd.shape[-1]} channels but the environment '
                    f'names {len(channels)}')
            # (2, lines, channels), lower first: rattlesnake's own plotting
            # reads [1] as upper and [0] as lower
            bands = {}
            for kind, variable in (('warning', 'specification_warning_matrix'),
                                   ('abort', 'specification_abort_matrix')):
                if variable in group.variables:
                    matrix = np.abs(np.asarray(group.variables[variable][()]))
                    bands[f'{kind}_lower'] = matrix[0]
                    bands[f'{kind}_upper'] = matrix[1]

            # The controller's target is a full matrix, and its cross
            # terms are read when they are real numbers: a file whose
            # off-diagonal is all NaN, or all zero, wrote placeholders
            # (Brandon, 2026-09-04: "probably usually just NaNs or 0"),
            # and a specification with only autos says so honestly —
            # the virtual point transform refuses it rather than
            # inventing the phase between the control channels.
            off_diagonal = cpsd[:, ~np.eye(len(channels), dtype=bool)]
            meaningful = bool(np.any(np.isfinite(off_diagonal)
                                     & (off_diagonal != 0)))
            records, response, reference, dims = [], [], [], []
            units, ref_units = [], []
            limits = {name: [] for name in bands}
            for i, (dof_i, (si, di, ui)) in enumerate(channels):
                for j, (dof_j, (sj, dj, uj)) in enumerate(channels):
                    if i != j and not (full_cpsd or meaningful):
                        continue
                    if i != j and not np.isfinite(cpsd[:, i, j]).any():
                        continue        # this one pair was never written
                    for name, matrix in bands.items():
                        # limits are per control channel, so a cross-spectral
                        # record has none of its own
                        limits[name].append(matrix[:, i] * si * sj if i == j
                                            else np.full(len(freq), np.nan))
                    records.append(cpsd[:, i, j] * si * sj)
                    response.append(dof_i)
                    reference.append(dof_j)
                    known = UNKNOWN not in (di, dj)
                    dims.append(
                        (f'{di}**2/frequency' if di == dj
                         else f'{di}*{dj}/frequency') if known else UNKNOWN)
                    units.append(ui if known else None)
                    ref_units.append(uj if known and ui != uj else None)
            out[f'{env_name}_specification'] = Specification(
                abscissa=freq,
                ordinate=np.array(records),
                response_dof=response,
                reference_dof=reference,
                ordinate_dim=dims,
                ordinate_unit=units,
                reference_unit=ref_units,
                **{name: np.array(values) for name, values in limits.items()},
            )

        # A sine environment's target is its tone set: a specifications
        # subgroup with one named group per tone, each a breakpoint
        # table with the per-segment sweep law. The stored table pads
        # its segment arrays to breakpoint length (the trailing entry is
        # dead — the leading-rate convention, pinned against the
        # controller's own trajectory record), and the band arrays are
        # (breakpoint, lower/upper, left/right, channel): the left and
        # right sides of a breakpoint may differ in principle, but no
        # file here has shown one that does, so a difference refuses by
        # name rather than being averaged into a band nobody wrote.
        for env_name, group in ds.groups.items():
            if 'specifications' not in group.groups:
                continue
            indices = np.asarray(
                group.variables['control_channel_indices'][()], dtype=int)
            scales = [scales_dims[ci][0] for ci in indices]
            control_dims = {scales_dims[ci][1] for ci in indices}
            control_units = {scales_dims[ci][2] for ci in indices}
            if len(control_dims) > 1:
                raise ValueError(
                    f'{env_name}: control channels mix quantities '
                    f'{sorted(control_dims)}; one sine specification '
                    'holds one')
            tones = []
            for tone_name, sub in group.groups['specifications'].groups.items():
                n = len(sub.variables['spec_frequency'][()])
                bands = {}
                for kind, variable in (('warning', 'spec_warning'),
                                       ('abort', 'spec_abort')):
                    if variable not in sub.variables:
                        continue
                    matrix = np.asarray(sub.variables[variable][()],
                                        dtype=float)
                    left, right = matrix[:, :, 0, :], matrix[:, :, 1, :]
                    if not np.allclose(left, right, equal_nan=True):
                        raise ValueError(
                            f'{env_name} tone {tone_name}: the {kind} '
                            'band differs between the left and right '
                            'sides of a breakpoint — never seen in a '
                            'real file, refused rather than averaged')
                    for side, curve in enumerate(('lower', 'upper')):
                        values = left[:, side, :] * np.asarray(scales)
                        if np.isfinite(values).any():
                            bands[f'{kind}_{curve}'] = values
                tones.append(SineTone(
                    name=tone_name,
                    start_time=float(sub.getncattr('start_time')),
                    frequency=sub.variables['spec_frequency'][()],
                    amplitude=(np.asarray(sub.variables['spec_amplitude'][()])
                               * np.asarray(scales)),
                    phase=sub.variables['spec_phase'][()],
                    segment_type=sub.variables['spec_sweep_type'][()][:n - 1],
                    segment_rate=sub.variables['spec_sweep_rate'][()][:n - 1],
                    **bands))
            dim = control_dims.pop()
            unit = (control_units.pop()
                    if len(control_units) == 1 else None)
            out[f'{env_name}_specification'] = SineSweepSpecification(
                tones=tones, response_dof=[dofs[ci] for ci in indices],
                ordinate_dim=dim, ordinate_unit=unit,
                comment=f'sine tones from the {env_name} environment')

        # A transient environment's target is a waveform, not a spectrum,
        # and rides the file as `control_signal` — (control channels,
        # samples) with no abscissa of its own, because the controller
        # plays it at the hardware's rate.
        for env_name, group in ds.groups.items():
            if 'control_signal' not in group.variables:
                continue
            signal = np.asarray(group.variables['control_signal'][()])
            indices = np.asarray(
                group.variables['control_channel_indices'][()], dtype=int)
            if signal.ndim != 2 or not len(indices):
                continue
            rate = _environment_rate(group, ds)
            if rate is None:
                continue
            records, response, dims, units = [], [], [], []
            for i, ci in enumerate(indices[:signal.shape[0]]):
                scale, dim, unit = scales_dims[ci]
                records.append(signal[i] * scale)
                response.append(dofs[ci])
                dims.append(dim)
                units.append(unit)
            out[f'{env_name}_specification'] = TransientSpecification(
                abscissa=np.arange(signal.shape[-1]) / rate,
                ordinate=np.array(records),
                response_dof=response,
                ordinate_dim=dims,
                ordinate_unit=units,
                comment='transient control signal')

    # a run also states its *control* channels, through its
    # specification: the requirement is written per control channel, so
    # a response channel whose DOF the specification names is a control
    # channel. The file's own statement, mapped rather than guessed —
    # and role-guarded, so a drive sharing a controlled DOF stays a
    # reference.
    named = {dof for obj in out.values()
             if isinstance(obj, (Specification, TransientSpecification,
                                 SineSweepSpecification))
             for dof in obj.response_dof}
    if named:
        roles = table.roles()
        for row, dof in enumerate(table.dof_strings()):
            if dof in named and roles[row] == 'response':
                table.set_cell('control', row, 'True')
    return out


def _environment_rate(group, ds):
    """The rate a control signal is played at.

    The environment's own sample rate when it says, the file's otherwise.
    A spectral save keeps only its environment group — root attributes
    and all — so the file's may simply not be there.
    """
    for holder, name in ((group, 'sysid_sample_rate'),
                         (group, 'sample_rate'), (ds, 'sample_rate')):
        if name in holder.ncattrs():
            return float(holder.getncattr(name))
    return None
