# The .vdyn file, laid out

A `.vdyn` is a standard HDF5 file with a documented layout inside; the
extension does not mean a new container. Rename one to `.h5` and
HDFView browses it, `h5py` walks it, and MATLAB's `h5read` reads every
dataset. This page writes the layout down so that you can read a file
without reading the source. In the code the layout is defined in one
module, [`visualdynamics.io.native`](api/visualdynamics.io.native.md),
and the frozen files under `testdata/vdyn_corpus/` hold each schema to
it.

## The version number

Every file stamps one root attribute:

    visualdynamics_schema = 1, 2 or 3   (integer)

The schema changes by addition. New datasets and attributes are
optional, with a default when they are absent, and existing ones are
never renamed or given a new meaning. Version 2 added the response
predictions (`predictions/`, below) and the report block that reads a
measurement against them; a reader that knows only version 1 would
meet a kind it cannot read, so a file holding either is stamped 2.
Version 3 added the run log extract (`run_log_extract/`, below), and a
file holding one is stamped 3. Every other file is stamped with the
earliest version that holds everything in it, 1 when it holds nothing
a later version added, so an older reader still opens it. The reader
follows three rules:

- a stamp newer than the reader knows is refused with a message saying
  the file was written by a newer Visual Dynamics and to update to open
  it; the file is never half loaded;
- a file with no stamp is refused as not a Visual Dynamics file, since
  every writer stamps;
- an older stamp always loads, and `tests/test_vdyn_corpus.py` opens
  the frozen files on every run.

Every `.vdyn` written by any release since the first (0.1.0a1) has
loaded in every later release, and the frozen corpus checks this on
every run. While Visual Dynamics is an alpha that is a practice, not a
promise: the `.vdyn` format may still change, and the promise that a
later version opens every earlier file starts with the first release
that is not an alpha. `.vdyn` is the project file, and it stays the
project file. The Engineering Sciences Common Data Format (`.h5`) is
imported and exported beside it, for exchanging data with other tools.

## Two kinds of file

A single object saved on its own is one group at the root, named for
its kind: `geometry/`, `data/`, `shapes/`, `report/`, `photos/`,
`matches/`, `channel_table/`, `sine_specification/`, `sine_levels/`,
`predictions/` or `run_log_extract/`.
The loader recognizes the file by the group it finds.

A whole project is an `objects/` container of numbered groups (`0000`,
`0001`, and so on, which keep the order). Each group has two
attributes, `name` (your name for the object, which may contain
anything HDF5 would otherwise read as a path) and `kind` (one of the
group names above), and holds the object's own layout. The project's
identity is in root attributes:

    test_name         the project's name
    project_type      '' or e.g. 'Modal Test', 'Random Vibration'
    active_geometry   '' or the name unlinked objects are read against
    links             JSON, the project's object groups: [{"members":
                      [names...], "role": null or "Basis", "name": ...},
                      ...]; "name" is present only for a named group.
                      Files older than the first release may also carry
                      a "side" key, which is read and never written
    provenance        JSON: {name: {"verb", "source", "params",
                      "state"}, ...}, how each derived object was
                      computed, with the settings fingerprint the
                      staleness badges compare against

A project whose data index holds anything also has one dataset at the
root (absent otherwise):

    data_index        JSON, one string: [{"name", "path", "size",
                      "sha256", "holds", ...}, ...], every file the
                      project rests on, in the order indexed. Optional
                      keys: "format", "role" ("referenced" or
                      "attached"), "source" (what added the entry),
                      "objects" (the names read from the file),
                      "indexed", "status" ("matches", "changed" or
                      "not found"), "checked" and "found" (the last
                      check), and "text" (an attached text file's
                      contents). A dataset rather than an attribute
                      because an attached file's text can pass the
                      64 KiB an HDF5 attribute holds

## Conventions

- Strings are UTF-8 (`h5py.string_dtype('utf-8')`), one dataset
  per field and one element per record. `''` means undefined: an
  ordinate unit of `''` is a channel whose units were never declared,
  and it reloads as undefined, not as dimensionless.
- Ragged arrays (element connectivity) are two datasets:
  `<name>_flat`, the values run together, and `<name>_offsets`, one
  longer than the count, so entry *i* is
  `flat[offsets[i]:offsets[i+1]]`.
- Optional fields are absent rather than given a sentinel value. A
  time history with no averaging has no `averaging_*` attributes, and
  a shape set that never carried complex modal damping has no
  `modal_damping` dataset. Absent fields take their defaults, which is
  how older files keep loading.
- Numeric datasets are whatever numpy held (float64, complex128,
  int64) at full precision. Nothing is narrowed.

## `geometry/`

    attrs   dimension, length_unit ('' = undeclared)
            mass_point, and mass and inertia when the rigid-body modes
              are to be mass normalized (only when set)
            view_eye, view_up      the view it opens on (only when set)
            solids                 JSON, the solids as their numbers
                                   were typed (only when there are any)
    dsets   node_id, node_xyz, node_def_cs, node_disp_cs, node_color,
            cs_id, cs_name, cs_type, cs_matrix,
            elem_id, elem_type, elem_color, elem_conn_flat / _offsets,
            elem_block, block_id, block_name
    block_properties/<group id>/   (only element groups given properties)
            attrs   material_name, youngs_modulus, density, poissons_ratio,
                    modulus_of_rigidity (when set), thickness (plates) or
                    section_name + section [area, iy, iz, j] (beams),
                    and section_shape + section_dimensions when the
                    section was built from a shape (fem.SHAPES; meters,
                    in the constructor's order),
                    orientation (beams, when set);
                    mass alone, kg, for a group of point masses;
                    stiffness, six values with NaN where a direction is
                    free, for a group of springs;
                    ground, the directions held ('X Y Z RX RY RZ'), for
                    a group of ground points (0.1.0a39 wrote True for
                    all six, which still reads)

The element groups are stored under their original names: `elem_block`,
`block_id` and `block_name` on disk are a geometry's `elem_group`,
`group_id` and `group_name`, and `block_properties` is its
`group_properties`. Files from every release read the same way.

A file written by 0.1.0a20 or earlier also holds `traceline_id`,
`traceline_color`, `traceline_desc` and `traceline_conn_flat /
_offsets`. The loader turns each traceline id into an element group of
two-node beam elements with no properties, named by its description,
and splits any element group that holds more than one element family.
The file is written back without them.

## `data/`, every measurement class

One layout serves TimeHistory, Spectrum, Frf, Psd, Srs, Coherence,
MultipleCoherence, Specification, ShockSpecification,
TransientSpecification, SineLevel and SineTarget. The concrete class is
stored by name, because a Specification has the PSD function type and
the function type alone cannot say which class to rebuild:

    attrs   function_type        UFF dataset 58 code
            data_class           the class name, e.g. 'Psd'
    dsets   abscissa             one shared axis
            ordinate             records × points, real or complex
            response_dof         strings, one per record
            ordinate_dim, comment, ordinate_unit, reference_unit,
            dimension_hint       strings, '' = undefined / not stated

Present only where they mean something:

    dsets   reference_dof     records that have a reference (FRF, coherence)
            block             which repeat a record is, for stacked captures
            limit_<name>      a specification's warning and abort lines
            bandwidth         band widths, where the bins are not the
                              midpoints between lines (octave bands)
            statistical_dof   a PSD's statistical degrees of freedom,
                              one per record, where it was computed
                              from a time history
            line_correlation  where it was computed from a time history,
                              how its window ties neighboring lines:
                              the correlation at 0, 1, 2, ... lines
                              apart, starting at 1, one for every record
            shocks            (start, duration) rows, where events were marked
            roles             strings, DOF, quantity and role in threes,
                              flat: each channel's role (reference,
                              response or monitor) where something has
                              set one. A channel not listed, or a file
                              without the dataset, takes the default for
                              its quantity (a force is a reference, a
                              motion a response, anything else a monitor)
            sine_seconds      a sine level's clock
            sine_floor, sine_below_floor   the noise floor beside each
                              sine reading, and where the tone was under it
    attrs   srs_q, srs_kind          what an SRS was computed with
            interpolation            how a specification reads between
                                     its lines
            band_constraints         JSON, how a specification's bands
                                     move together when edited
            sine_tone, sine_onset, sine_drift_hz   which tone a sine
                                     level belongs to, and where its
                                     sweep began
            sine_cycles (when set), sine_target_db, sine_refine,
            sine_chosen (when set)   the sine extraction setting
            scale_db                 a comparison scale set by hand only;
                                     automatic scaling is never saved
            narrowbands              JSON, the narrow bands of random a
                                     mixed test lays over this floor
                                     (each a center, a width, a level and
                                     a sweep), when it has any
            tone_exclusion           the width, Hz, left out of this
                                     floor either side of a mixed test's
                                     tones, when set
            run_identity             JSON, a time history's run: axis,
                                     run, level_db, start and comment
                                     where known, "sources" (where each
                                     came from) and "refused" (what was
                                     not read, and why); absent when
                                     nothing has said. The level the
                                     comparison detects is never saved
            averaging_frame_length, _overlap, _window, _frames, _start,
            _detrend, _start_set, and _window_parameter when the window
            takes one
            filtering_low, _high (each when set), _order   the filter
                                     reading
            truncation_start, _stop  the truncate reading, when set

## `shapes/`

    attrs   mass_unit ('' = undeclared), unscaled (no drive point, so
            the modal masses are a convention, not a measurement)
    dsets   frequency, damping, shape_matrix (modes × DOFs, complex
            where a source carried complex modes), modal_mass
            (complex where measured), coordinate, comment, description
            modal_damping        only where a source carried one

## `channel_table/`

    attrs   column_order         every column, in the file's order
    dsets   channel, range       integers (the schema's 'int' columns)
            <every other column> strings, verbatim (node included: a
                                 node nobody recorded is blank, since 0
                                 cannot stand for "not stated")

## `report/`, `photos/`, `matches/`

    report/   attrs: title, marking, marking_color ('ink' | 'red';
              marking '' = no strips);  dsets: blocks (one JSON string)
    photos/   dsets: names, formats, image_0000, ... (each photograph
              file's own bytes, verbatim, never re-encoded)
    matches/  attrs: first, second, first_geometry, second_geometry
              dsets: pairs (n × 2 mode indices), macs

## `sine_specification/`, `sine_levels/`

    sine_specification/  attrs: ordinate_dim, ordinate_unit, comment
                         dsets: response_dof, tone_order
                         tones/<k>/: attrs name, start_time; dsets
                         frequency, amplitude, phase, segment_type,
                         segment_rate, and any warning or abort limits
                         (the file's own breakpoints, per tone)
    sine_levels/         attrs: sine_cycles (when set)
                         dsets: tone_order; tones/<k>/ each an
                         ordinary data/ layout (the set only groups
                         them)

## `predictions/` (schema 2)

    predictions/  attrs: comment, count
                  <k>/: attrs record (one JSON string: channel,
                  quantity, unit, level_db, reference, source, customer,
                  frequency_uncertainty, level_uncertainty, and peaks,
                  each [frequency, level, Q, band] with null where not
                  stated); dsets curve_frequency, curve_values (a
                  predicted curve, when there is one)

A report's blocks may include one of kind `prediction`, naming the
predictions (`source`) and the measurement (`measured`) with the level
it was run at (`level`, dB relative to full level, or null).

## `run_log_extract/` (schema 3)

    run_log_extract/  attrs: path, test, upto, taken, log_size, after,
                      after_in_test, problems (a JSON list of strings)
                      dset log (one UTF-8 string: the snapshot, the
                      run log's own JSON Lines)

One test of a run log (`.vdlog`), frozen at entry `upto`: `log` holds
the log's header line and the lines of the entries kept, exactly as the
log file had them, so it reads with the run log's own reader for its
version and its chains verify. `path` is where the log was, relative to
the project file (with forward slashes) when the log was in the
project file's folder or one beneath it, absolute otherwise. `taken` is
when the snapshot was taken (ISO 8601 with its offset), `log_size` the
log file's size in bytes then, `after` and `after_in_test` how many
entries the file held after `upto` then, in all and of the test, and
`problems` the lines before `upto` skipped as not entries.

## Reading one without Visual Dynamics

Because the layout is documented HDF5, this works:

```python
import h5py

with h5py.File('modal.vdyn') as f:
    for key in sorted(f['objects']):
        group = f['objects'][key]
        print(group.attrs['name'], group.attrs['kind'])
    frf = f['objects/0003']                    # say it's the FRF
    H = frf['ordinate'][()]                    # records × frequencies
    freq = frf['abscissa'][()]
```

MATLAB reads the same file with `h5info('modal.vdyn')` and `h5read`.
Nothing in a `.vdyn` needs this package to get it back out. If you
would rather `load` structs in MATLAB than walk datasets, the same
layout can be written as a `.mat` file, with `project.save('modal.mat')`
or *Save Project As…* in the app, and read back from one. The
[export page](export.md) says what MATLAB spells differently.
