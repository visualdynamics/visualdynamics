"""Native .vdyn file format: HDF5.

A single object is one group named for its kind at the root. A whole test
is an `objects/` container of numbered groups — number for order, `name`
and `kind` as attributes, so a user's name is free to contain anything
h5py would read as structure — plus the test's own name and active
geometry as root attributes. Ragged connectivity is stored as a flat
array plus offsets.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..project import Project

if TYPE_CHECKING:                                    # pragma: no cover
    import h5py

    from ..core.channel_table import ChannelTable
    from ..core.data import DataArray
    from ..core.geometry import Geometry
    from ..core.matches import MatchedModes
    from ..core.photos import Photos
    from ..core.report import Report
    from ..core.shapes import ShapeSet

SCHEMA_VERSION = 1


def _write_ragged(group: h5py.Group, name: str,
                  arrays: Sequence[np.ndarray]) -> None:
    flat = (np.concatenate(arrays) if arrays
            else np.empty(0, dtype=np.int64))
    offsets = np.cumsum([0] + [len(a) for a in arrays])
    group.create_dataset(f'{name}_flat', data=flat)
    group.create_dataset(f'{name}_offsets', data=offsets)


def _read_ragged(group: h5py.Group, name: str) -> list[np.ndarray]:
    flat = group[f'{name}_flat'][()]
    offsets = group[f'{name}_offsets'][()]
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def _write_strings(group: h5py.Group, name: str,
                   strings: Sequence[str]) -> None:
    import h5py
    group.create_dataset(name, data=strings, dtype=h5py.string_dtype('utf-8'))


def _read_strings(group: h5py.Group, name: str) -> list[str]:
    return [s.decode('utf-8') for s in group[name][()]]


def save_geometry(geom: Geometry, group: h5py.Group) -> None:
    group.attrs['dimension'] = geom.dimension
    group.attrs['length_unit'] = geom.length_unit or ''
    for name in ('node_id', 'node_xyz', 'node_def_cs', 'node_disp_cs', 'node_color',
                 'cs_id', 'cs_type', 'cs_matrix',
                 'traceline_id', 'traceline_color',
                 'elem_id', 'elem_type', 'elem_color', 'elem_block',
                 'block_id'):
        group.create_dataset(name, data=getattr(geom, name))
    _write_strings(group, 'cs_name', geom.cs_name)
    _write_strings(group, 'block_name', geom.block_name)
    _write_strings(group, 'traceline_desc', geom.traceline_desc)
    _write_ragged(group, 'traceline_conn', geom.traceline_conn)
    _write_ragged(group, 'elem_conn', geom.elem_conn)
    # the rigid-body settings, when set — the point always, the mass
    # and inertia only when the set is to be mass-normalized, so an
    # unscaled setting and a file from before the setting existed read
    # back the same way (absent, not sentinel-valued)
    properties = getattr(geom, 'mass_properties', None)
    if properties is not None:
        group.attrs['mass_point'] = np.asarray(properties.point)
        if properties.scaled:
            group.attrs['mass'] = properties.mass
            group.attrs['inertia'] = np.asarray(properties.inertia)


def _load_mass_properties(group):
    if 'mass_point' not in group.attrs:
        return None
    from ..core.rigid import MassProperties

    return MassProperties(
        tuple(group.attrs['mass_point']),
        mass=float(group.attrs['mass']) if 'mass' in group.attrs else None,
        inertia=(tuple(group.attrs['inertia']) if 'inertia' in group.attrs
                 else None))


def load_geometry(group: h5py.Group) -> Geometry:
    from ..core.geometry import Geometry

    data = {name: group[name][()] for name in (
        'node_id', 'node_xyz', 'node_def_cs', 'node_disp_cs', 'node_color',
        'cs_id', 'cs_type', 'cs_matrix',
        'traceline_id', 'traceline_color',
        'elem_id', 'elem_type', 'elem_color', 'elem_block', 'block_id')}
    data['block_name'] = _read_strings(group, 'block_name')
    data['cs_name'] = _read_strings(group, 'cs_name')
    data['traceline_desc'] = _read_strings(group, 'traceline_desc')
    data['traceline_conn'] = _read_ragged(group, 'traceline_conn')
    data['elem_conn'] = _read_ragged(group, 'elem_conn')
    data['length_unit'] = group.attrs.get('length_unit', '') or None
    geometry = Geometry(**data)
    geometry.mass_properties = _load_mass_properties(group)
    return geometry


def save_data(data: DataArray, group: h5py.Group) -> None:
    from ..core.data import DataArray  # noqa: F401 (documented schema owner)

    group.attrs['function_type'] = data.function_type
    group.create_dataset('abscissa', data=data.abscissa)
    group.create_dataset('ordinate', data=data.ordinate)
    _write_strings(group, 'response_dof', data.response_dof)
    if data.reference_dof is not None:
        _write_strings(group, 'reference_dof', data.reference_dof)
    # which repeat a record is, where records are the same measurement more
    # than once; absent entirely when they are not
    if data.block is not None:
        _write_strings(group, 'block', data.block)
    _write_strings(group, 'ordinate_dim', data.ordinate_dim)
    _write_strings(group, 'comment', data.comment)
    # '' marks a record whose units have not been defined
    _write_strings(group, 'ordinate_unit',
                   [u or '' for u in data.ordinate_unit])
    _write_strings(group, 'reference_unit',
                   [u or '' for u in data.reference_unit])
    # what the source said the quantity was, where it said that but gave no
    # scale; '' where it said nothing
    _write_strings(group, 'dimension_hint',
                   [h or '' for h in data.dimension_hint])
    # the concrete class: a Specification shares the PSD function type, so
    # the code alone cannot say which to rebuild
    group.attrs['data_class'] = type(data).__name__
    # what an SRS was worked out at. Two numbers that change what the
    # curve means, so they travel with it rather than being assumed
    if getattr(data, 'q', None) is not None:
        group.attrs['srs_q'] = float(data.q)
        group.attrs['srs_kind'] = str(data.kind)
    # which specification tone an extracted sine level answers to, and
    # where its sweep was found in the recording
    if getattr(data, 'tone', ''):
        group.attrs['sine_tone'] = str(data.tone)
        group.attrs['sine_onset'] = float(data.onset)
        if getattr(data, 'seconds', None) is not None:
            group.create_dataset('sine_seconds', data=data.seconds)
    # how a spectrum's values are read between the points. It travels
    # because it cannot be worked out again from the numbers: a
    # specification computed from a record and one written by hand can
    # both arrive as several hundred evenly spaced lines — Rattlesnake
    # writes its own on the control lines — and only the first is a
    # density. Left behind, a reopened project drew four thousand lines
    # of one as a power law through themselves.
    if getattr(data, 'interpolation', None) is not None:
        group.attrs['interpolation'] = str(data.interpolation)
    # a specification's band constraints — symmetric, uniform, per
    # channel — are how its sheet moves the bands, not derivable from
    # limits that happen to mirror (2026-09-06)
    if getattr(data, 'band_constraints', None):
        import json

        group.attrs['band_constraints'] = json.dumps(data.band_constraints)
    # a held comparison scale is the user's judgment, not a derivable
    # number — auto-detected scaling is None here and re-detected on
    # every comparison, so only a hand-set value travels
    if getattr(data, 'scale_db', None) is not None:
        group.attrs['scale_db'] = int(data.scale_db)
    for name, values in getattr(data, 'limits', {}).items():
        group.create_dataset(f'limit_{name}', data=values)
    # how a time history is to be cut into frames, when something has
    # said — from the file it was imported from, or from the averaging
    # view. Five attributes rather than a group: it is five numbers.
    averaging = getattr(data, 'averaging', None)
    if averaging is not None:
        group.attrs['averaging_frame_length'] = averaging.frame_length
        group.attrs['averaging_overlap'] = averaging.overlap
        group.attrs['averaging_window'] = averaging.window
        group.attrs['averaging_frames'] = averaging.frames
        group.attrs['averaging_start'] = averaging.start
        group.attrs['averaging_detrend'] = averaging.detrend
        if averaging.window_parameter is not None:
            group.attrs['averaging_window_parameter'] = \
                averaging.window_parameter
    # the low-pass the history is read through, when something has
    # said — two numbers, so two attributes, like the averaging's five
    filtering = getattr(data, 'filtering', None)
    if filtering is not None:
        # a pass-band edge that is not set is *absent*, matching the
        # class: None is the meaning, not a value to spell
        if filtering.low is not None:
            group.attrs['filtering_low'] = filtering.low
        if filtering.high is not None:
            group.attrs['filtering_high'] = filtering.high
        group.attrs['filtering_order'] = filtering.order
    truncation = getattr(data, 'truncation', None)
    if truncation is not None:
        group.attrs['truncation_start'] = truncation.start
        group.attrs['truncation_stop'] = truncation.stop
    # where the shocks are, when something has said. One row of
    # (start, duration) per event rather than attributes: there is no
    # fixed number of them
    # the width of each band, for a spectrum whose bins are not the
    # midpoints between its own lines
    if getattr(data, 'bandwidth', None) is not None:
        group.create_dataset('bandwidth', data=np.asarray(data.bandwidth))
    found = getattr(data, 'shocks', None)
    if found:
        group.create_dataset('shocks', data=np.array(
            [[s.start, s.duration] for s in found], dtype=float))


def _load_shocks(group):
    if 'shocks' not in group:
        return None
    from ..core.shocks import Shock

    return tuple(Shock(float(start), float(duration))
                 for start, duration in group['shocks'][()])


def _load_averaging(group):
    if 'averaging_frame_length' not in group.attrs:
        return None
    from ..core.averaging import Averaging

    return Averaging(
        frame_length=int(group.attrs['averaging_frame_length']),
        overlap=float(group.attrs['averaging_overlap']),
        window=str(group.attrs['averaging_window']),
        frames=int(group.attrs['averaging_frames']),
        start=float(group.attrs['averaging_start']),
        # absent in every file written before 2026-08-29: the
        # default is the behavior those files were computed with
        detrend=str(group.attrs.get('averaging_detrend', 'none')),
        window_parameter=(
            float(group.attrs['averaging_window_parameter'])
            if 'averaging_window_parameter' in group.attrs
            else None))


def load_data(group: h5py.Group) -> DataArray:
    from ..core.data import NAMED_CLASSES, class_for_function_type

    cls = NAMED_CLASSES.get(group.attrs.get('data_class'))
    if cls is None:
        cls = class_for_function_type(group.attrs['function_type'])
    limits = {key[len('limit_'):]: group[key][()]
              for key in group if key.startswith('limit_')}
    if 'bandwidth' in group:
        limits['bandwidth'] = group['bandwidth'][()]
    if 'srs_q' in group.attrs:
        limits['q'] = float(group.attrs['srs_q'])
        limits['kind'] = str(group.attrs['srs_kind'])
    if 'sine_tone' in group.attrs:
        limits['tone'] = str(group.attrs['sine_tone'])
        limits['onset'] = float(group.attrs['sine_onset'])
        if 'sine_seconds' in group:
            limits['seconds'] = group['sine_seconds'][()]
    data = cls(
        **limits,
        abscissa=group['abscissa'][()],
        ordinate=group['ordinate'][()],
        response_dof=_read_strings(group, 'response_dof'),
        reference_dof=(_read_strings(group, 'reference_dof')
                       if 'reference_dof' in group else None),
        block=_read_strings(group, 'block') if 'block' in group else None,
        ordinate_dim=_read_strings(group, 'ordinate_dim'),
        comment=_read_strings(group, 'comment'),
        ordinate_unit=_read_strings(group, 'ordinate_unit'),
        reference_unit=_read_strings(group, 'reference_unit'),
        dimension_hint=_read_strings(group, 'dimension_hint'),
    )
    if 'interpolation' in group.attrs:
        data.interpolation = str(group.attrs['interpolation'])
    if 'band_constraints' in group.attrs:
        import json

        data.band_constraints = json.loads(str(group.attrs['band_constraints']))
    if 'scale_db' in group.attrs:
        data.scale_db = int(group.attrs['scale_db'])
    averaging = _load_averaging(group)
    if averaging is not None:
        data.averaging = averaging
    if 'filtering_low' in group.attrs or 'filtering_high' in group.attrs:
        from ..core.filters import Filtering

        low = group.attrs.get('filtering_low')
        high = group.attrs.get('filtering_high')
        data.filtering = Filtering(
            low=None if low is None else float(low),
            high=None if high is None else float(high),
            order=int(group.attrs['filtering_order']))
    if 'truncation_start' in group.attrs:
        from ..core.truncate import Truncation

        data.truncation = Truncation(
            float(group.attrs['truncation_start']),
            float(group.attrs['truncation_stop']))
    found = _load_shocks(group)
    if found is not None:
        data.shocks = found
    return data


def save_channel_table(table: ChannelTable, group: h5py.Group) -> None:
    group.attrs['column_order'] = table.column_names
    for name in table.column_names:
        # the typed columns decide the storage: node joined channel as an
        # integer when the schema grew rules, and h5py refuses to write
        # int64 through a string dtype rather than coercing it
        if table.COLUMNS[name].kind == 'int':
            group.create_dataset(name, data=table[name])
        else:
            _write_strings(group, name, list(table[name]))


def load_channel_table(group: h5py.Group) -> ChannelTable:
    from ..core.channel_table import ChannelTable

    columns = {}
    for name in group.attrs['column_order']:
        # read by what is stored rather than by the column specs, so
        # the storage rule lives once, in the writer — and a column the
        # schema has since dropped still reads (the constructor is what
        # decides which columns survive)
        data = group[name]
        columns[name] = (data[()] if data.dtype.kind in 'iuf'
                         else _read_strings(group, name))
    return ChannelTable(columns)


def save_shapes(shapes: ShapeSet, group: h5py.Group) -> None:
    group.attrs['mass_unit'] = shapes.mass_unit or ''
    group.attrs['unscaled'] = bool(getattr(shapes, 'unscaled', False))
    group.create_dataset('frequency', data=shapes.frequency)
    group.create_dataset('damping', data=shapes.damping)
    group.create_dataset('shape_matrix', data=shapes.shape_matrix)
    group.create_dataset('modal_mass', data=shapes.modal_mass)
    # written only when a source carried one, so an old file and a set
    # that never had it read back identically
    if shapes.modal_damping is not None:
        group.create_dataset('modal_damping', data=shapes.modal_damping)
    _write_strings(group, 'coordinate', shapes.coordinate)
    _write_strings(group, 'comment', shapes.comment)
    _write_strings(group, 'description', shapes.description)


def load_shapes(group: h5py.Group) -> ShapeSet:
    from ..core.shapes import ShapeSet

    return ShapeSet(
        frequency=group['frequency'][()],
        damping=group['damping'][()],
        coordinate=_read_strings(group, 'coordinate'),
        shape_matrix=group['shape_matrix'][()],
        modal_mass=group['modal_mass'][()],
        comment=_read_strings(group, 'comment'),
        description=_read_strings(group, 'description'),
        mass_unit=group.attrs['mass_unit'] or None,
        unscaled=bool(group.attrs['unscaled']),
        modal_damping=(group['modal_damping'][()]
                       if 'modal_damping' in group else None),
    )


def save_report(report: Report, group: h5py.Group) -> None:
    import json

    import h5py

    group.attrs['title'] = report.title
    group.attrs['marking'] = report.marking
    group.attrs['marking_color'] = report.marking_color
    group.create_dataset('blocks', data=json.dumps(report.blocks),
                         dtype=h5py.string_dtype('utf-8'))


def load_report(group: h5py.Group) -> Report:
    import json

    from ..core.report import Report

    blocks = group['blocks'][()]
    if isinstance(blocks, bytes):
        blocks = blocks.decode('utf-8')
    return Report(title=str(group.attrs['title']),
                  blocks=json.loads(blocks),
                  marking=str(group.attrs['marking']),
                  marking_color=str(group.attrs['marking_color']))


def save_photos(photos: Photos, group: h5py.Group) -> None:
    _write_strings(group, 'names', photos.names)
    _write_strings(group, 'formats', photos.formats)
    for i, image in enumerate(photos.images):
        # the file's own bytes, verbatim — no decode, no re-encode
        group.create_dataset(f'image_{i:04d}',
                             data=np.frombuffer(image, dtype=np.uint8))


def load_photos(group: h5py.Group) -> Photos:
    from ..core.photos import Photos

    names = _read_strings(group, 'names')
    return Photos(names=names,
                  formats=_read_strings(group, 'formats'),
                  images=[group[f'image_{i:04d}'][()].tobytes()
                          for i in range(len(names))])


def save_matches(matches: MatchedModes, group: h5py.Group) -> None:
    group.attrs['first'] = matches.first
    group.attrs['second'] = matches.second
    group.attrs['first_geometry'] = matches.first_geometry or ''
    group.attrs['second_geometry'] = matches.second_geometry or ''
    group.create_dataset('pairs', data=np.asarray(
        matches.pairs, dtype=np.int64).reshape(-1, 2))
    group.create_dataset('macs', data=np.asarray(
        matches.macs, dtype=np.float64))


def load_matches(group: h5py.Group) -> MatchedModes:
    from ..core.matches import MatchedModes

    return MatchedModes(
        group.attrs['first'], group.attrs['second'],
        pairs=group['pairs'][()].tolist(),
        macs=group['macs'][()].tolist(),
        first_geometry=group.attrs['first_geometry'] or None,
        second_geometry=group.attrs['second_geometry'] or None)


def save_sine_specification(spec, group) -> None:
    """One subgroup per tone, the file's own shape: breakpoints with
    the per-segment sweep law, bands where they exist."""
    _write_strings(group, 'response_dof', spec.response_dof)
    group.attrs['ordinate_dim'] = spec.ordinate_dim
    group.attrs['ordinate_unit'] = spec.ordinate_unit or ''
    group.attrs['comment'] = spec.comment
    _write_strings(group, 'tone_order', [tone.name for tone in spec.tones])
    tones = group.create_group('tones')
    for k, tone in enumerate(spec.tones):
        # index-named subgroups: tone names are user text and h5py
        # group names cannot hold a '/', so the order array holds the
        # names and the groups hold the numbers
        sub = tones.create_group(str(k))
        sub.attrs['name'] = tone.name
        sub.attrs['start_time'] = tone.start_time
        sub.create_dataset('frequency', data=tone.frequency)
        sub.create_dataset('amplitude', data=tone.amplitude)
        sub.create_dataset('phase', data=tone.phase)
        sub.create_dataset('segment_type', data=tone.segment_type)
        sub.create_dataset('segment_rate', data=tone.segment_rate)
        for name, values in tone.limits.items():
            sub.create_dataset(name, data=values)


def load_sine_specification(group):
    from ..core.sine import SineSweepSpecification, SineTone

    tones = []
    for k, name in enumerate(_read_strings(group, 'tone_order')):
        sub = group['tones'][str(k)]
        limits = {limit: sub[limit][()]
                  for limit in SineTone.LIMITS if limit in sub}
        tones.append(SineTone(
            name=name, start_time=float(sub.attrs['start_time']),
            frequency=sub['frequency'][()],
            amplitude=sub['amplitude'][()], phase=sub['phase'][()],
            segment_type=sub['segment_type'][()],
            segment_rate=sub['segment_rate'][()], **limits))
    return SineSweepSpecification(
        tones=tones, response_dof=_read_strings(group, 'response_dof'),
        ordinate_dim=str(group.attrs['ordinate_dim']),
        ordinate_unit=str(group.attrs['ordinate_unit']) or None,
        comment=str(group.attrs['comment']))


def save_sine_levels(levels, group) -> None:
    """One subgroup per tone, each the ordinary data layout — the set
    is grouping, not a new format."""
    _write_strings(group, 'tone_order',
                   [level.tone for level in levels.levels])
    tones = group.create_group('tones')
    for k, level in enumerate(levels.levels):
        save_data(level, tones.create_group(str(k)))


def load_sine_levels(group):
    from ..core.sine import SineLevelSet

    return SineLevelSet([load_data(group['tones'][str(k)])
                         for k in range(len(
                             _read_strings(group, 'tone_order')))])


def _savers():
    from ..core.channel_table import ChannelTable
    from ..core.data import DataArray
    from ..core.geometry import Geometry
    from ..core.matches import MatchedModes
    from ..core.photos import Photos
    from ..core.report import Report
    from ..core.shapes import ShapeSet
    from ..core.sine import SineLevelSet, SineSweepSpecification

    return [(Geometry, 'geometry', save_geometry),
            (SineSweepSpecification, 'sine_specification',
             save_sine_specification),
            (SineLevelSet, 'sine_levels', save_sine_levels),
            (DataArray, 'data', save_data),
            (ShapeSet, 'shapes', save_shapes),
            (Report, 'report', save_report),
            (Photos, 'photos', save_photos),
            (MatchedModes, 'matches', save_matches),
            (ChannelTable, 'channel_table', save_channel_table)]


_LOADERS = {'geometry': load_geometry,
            'sine_specification': load_sine_specification,
            'sine_levels': load_sine_levels,
            'data': load_data,
            'shapes': load_shapes,
            'report': load_report,
            'photos': load_photos,
            'matches': load_matches,
            'channel_table': load_channel_table}


def _saver_for(obj):
    for cls, group_name, saver in _savers():
        if isinstance(obj, cls):
            return group_name, saver
    raise TypeError(f"Don't know how to save {type(obj).__name__}")


def _visualdynamics_path(path: str | os.PathLike) -> str:
    path = str(path)
    return path if path.endswith('.vdyn') else path + '.vdyn'


def save(obj: Any, path: str | os.PathLike) -> None:
    """Save a visualdynamics object to a .vdyn (HDF5) file."""
    import h5py

    with h5py.File(_visualdynamics_path(path), 'w') as f:
        save_into(obj, f)


def save_into(obj: Any, f: h5py.File) -> None:
    """Write one object into an open, empty HDF5 file — the whole of
    `save` but the opening, so another container (`io.matlab`) can run
    the same writer into memory and carry the tree away."""
    group_name, saver = _saver_for(obj)
    f.attrs['visualdynamics_schema'] = SCHEMA_VERSION
    saver(obj, f.create_group(group_name))


# A whole test read back from one file is a Project: {name: object} in
# saved order, plus what the test carried — its name, active geometry,
# type and links. The alias stays because a saved test is exactly what
# `Project` means, and callers testing `isinstance(x, TestContents)`
# are asking "is this a project, not a foreign reader's dict?"
TestContents = Project


def save_test(path: str | os.PathLike, name: str,
              objects: Mapping[str, Any],
              active_geometry: str | None = None,
              project_type: str | None = None,
              links: Sequence[Mapping[str, Any]] | None = None,
              provenance: Mapping[str, Any] | None = None) -> None:
    """Save a whole test — every named object — to one .vdyn file.

    Objects go in numbered groups with the name as an attribute, so a name
    is free to contain anything h5py would read as structure. `links` is
    the explicit association groups, lists of object names.
    """
    import h5py

    with h5py.File(_visualdynamics_path(path), 'w') as f:
        save_test_into(f, name, objects, active_geometry, project_type,
                       links, provenance)


def save_test_into(f: h5py.File, name: str, objects: Mapping[str, Any],
                   active_geometry: str | None = None,
                   project_type: str | None = None,
                   links: Sequence[Mapping[str, Any]] | None = None,
                   provenance: Mapping[str, Any] | None = None) -> None:
    """`save_test` into an open, empty HDF5 file (see `save_into`)."""
    f.attrs['visualdynamics_schema'] = SCHEMA_VERSION
    f.attrs['test_name'] = name
    f.attrs['active_geometry'] = active_geometry or ''
    f.attrs['project_type'] = project_type or ''
    import json
    f.attrs['links'] = json.dumps(links or [])
    # how each derived object was computed, for the staleness
    # badges — settings fingerprints, so they survive the file
    f.attrs['provenance'] = json.dumps(provenance or {})
    container = f.create_group('objects')
    for i, (obj_name, obj) in enumerate(objects.items()):
        kind, saver = _saver_for(obj)
        group = container.create_group(f'{i:04d}')
        group.attrs['name'] = obj_name
        group.attrs['kind'] = kind
        saver(obj, group)


def load(path: str | os.PathLike,
         progress: Callable[[int, int], None] | None = None) -> Any:
    """Load a .vdyn file: the object it contains, or a whole test.

    `progress` is called as (objects loaded, objects in the file) —
    once up front with 0 and once per object — because a project file
    is minutes of someone's day and the reader is the only thing that
    knows how far along it is. A single-object file reports nothing:
    one object is one step, and a bar with one step is a light bulb.
    """
    import h5py

    with h5py.File(path, 'r') as f:
        return load_from(f, progress, str(path))


def load_from(f: h5py.File, progress: Callable[[int, int], None] | None = None,
              path: str = '') -> Any:
    """`load` from an open HDF5 file (see `save_into`); `path` names
    the file in the errors."""
    # A newer stamp means a newer Visual Dynamics wrote fields this
    # reader has no idea exist, and half-loading someone's project
    # quietly is worse than telling them to update. A missing stamp
    # means the file is not ours at all — every writer stamps.
    if 'visualdynamics_schema' not in f.attrs:
        raise ValueError(f'{path} is not a Visual Dynamics file '
                         '(no schema stamp)')
    written = int(f.attrs['visualdynamics_schema'])
    if written > SCHEMA_VERSION:
        raise ValueError(
            f'{path} was written by a newer Visual Dynamics '
            f'(schema {written}; this build reads up to '
            f'{SCHEMA_VERSION}). Update to open it.')
    if 'objects' in f:
        objects = {}
        keys = sorted(f['objects'])
        if progress is not None:
            progress(0, len(keys))
        for done, key in enumerate(keys, start=1):
            group = f['objects'][key]
            objects[group.attrs['name']] = (
                _LOADERS[group.attrs['kind']](group))
            if progress is not None:
                progress(done, len(keys))
        import json
        return Project(f.attrs['test_name'], objects,
                       f.attrs['active_geometry'] or None,
                       f.attrs['project_type'] or None,
                       json.loads(f.attrs['links']),
                       provenance=json.loads(f.attrs['provenance']))
    for group_name, loader in _LOADERS.items():
        if group_name in f:
            return loader(f[group_name])
    raise ValueError(f"No recognized content in {path}")
