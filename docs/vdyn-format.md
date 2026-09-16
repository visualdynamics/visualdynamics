# The .vdyn file, laid out

A `.vdyn` is a **standard HDF5 file** — the extension is a promise
about the layout inside, not a new container. Rename one to `.h5` and
HDFView browses it, `h5py` walks it, MATLAB's `h5read` reads every
dataset. This page is the layout written down, so that claim is
checkable without reading source; the schema's one owner in code is
[`visualdynamics.io.native`](api/visualdynamics.io.native.md), and the
frozen files under `testdata/vdyn_corpus/` hold each schema era to it.

## The one version number

Every file stamps a root attribute:

    visualdynamics_schema = 1        (integer)

The schema evolves **by addition**: new datasets and attributes are
optional with a default on read, existing ones are never renamed or
repurposed, and the number moves only if a field's *meaning* would
have to change — which has not happened yet. The reader's contract:

- a **newer** stamp than the reader knows is refused loudly ("written
  by a newer Visual Dynamics — update to open it"), never
  half-loaded;
- a **missing** stamp is refused as not a Visual Dynamics file at all
  — every writer stamps;
- an **older** stamp always loads — every reader keeps its past, and
  `tests/test_vdyn_corpus.py` opens every frozen era on every run.

Before the first public release the "keeps its past" half was a
soft promise, and the format used it: pre-release eras were dropped
(2026-08-23), so the reader holds exactly schema 1 with no fallbacks
for earlier spellings. Since the first release (0.1.0a1, 2026-09-14)
the contract is hard *for as long as the format lives* — and that is
the caveat this page has to state: **`.vdyn` is provisional, an
alpha-only format**. The next alpha saves projects in the Engineering
Sciences Common Data Format (`.escdf`), a public standard, and still
opens `.vdyn`; the first release that is not an alpha will not. Open
each `.vdyn` once in that alpha and save it again. Until then every
`.vdyn` written by any alpha loads in every later alpha.

## Two shapes of file

**A single object** saved on its own is one group at the root, named
for its kind: `geometry/`, `data/`, `shapes/`, `report/`, `photos/`,
`matches/`, `channel_table/`, `sine_specification/` or `sine_levels/`.
The loader recognises the file by which group it finds.

**A whole project** is an `objects/` container of numbered groups —
`0000`, `0001`, … for order — each carrying two attributes, `name`
(the user's name for the object, free to contain anything HDF5 would
read as path structure) and `kind` (one of the group names above),
with the object's own layout inside. The project's identity rides as
root attributes:

    test_name         the project's name
    project_type      '' or e.g. 'Modal Test', 'Random Vibration'
    active_geometry   '' or the name unlinked objects read against
    links             JSON: [{"members": [names...], "role": null or
                      "Basis"}, ...] (files before 2026-08-30 carried a
                      "side" key too; it is read and never written)
    provenance        JSON: {name: {"verb", "source", "params",
                      "state"}, ...} — how each derived object was
                      computed, with the settings fingerprint the
                      staleness badges compare against

## Conventions, everywhere

- **Strings** are UTF-8 (`h5py.string_dtype('utf-8')`), one dataset
  per field, one element per record. `''` marks *undefined* — an
  ordinate unit of `''` is a channel whose units were never declared,
  and it reloads as undefined, not as dimensionless.
- **Ragged arrays** (traceline and element connectivity) are two
  datasets: `<name>_flat`, the values run together, and
  `<name>_offsets`, one longer than the count, so entry *i* is
  `flat[offsets[i]:offsets[i+1]]`.
- **Optional fields are absent, not sentinel-valued.** A time history
  with no averaging has no `averaging_*` attributes; a shape set that
  never carried complex modal damping has no `modal_damping` dataset.
  Absent-with-a-default is what keeps every old file loading.
- Numeric datasets are whatever numpy held — float64, complex128,
  int64 — at full precision. Nothing narrows.

## `geometry/`

    attrs   dimension, length_unit ('' = undeclared)
    dsets   node_id, node_xyz, node_def_cs, node_disp_cs, node_color,
            cs_id, cs_name, cs_type, cs_matrix,
            traceline_id, traceline_color, traceline_desc,
            traceline_conn_flat / _offsets,
            elem_id, elem_type, elem_color, elem_conn_flat / _offsets,
            elem_block, block_id, block_name

## `data/` — every measurement class

One layout serves TimeHistory, Spectrum, Frf, Psd, Specification,
Coherence, MultipleCoherence, OctaveBandPsd, Srs — the concrete class
is named, because a Specification shares the PSD function type and
the code alone could not say which to rebuild:

    attrs   function_type        UFF dataset 58 code
            data_class           the class name, e.g. 'Psd'
    dsets   abscissa             one shared axis
            ordinate             records × points, real or complex
            response_dof         strings, one per record
            ordinate_dim, comment, ordinate_unit, reference_unit,
            dimension_hint       strings, '' = undefined / unsaid

Present only where they mean something:

    reference_dof     records that have a reference (FRF, coherence)
    block             which repeat a record is, for stacked captures
    limit_<name>      a specification's warning/abort lines
    bandwidth         octave-band widths, where bins are not midpoints
    shocks            (start, duration) rows, where events were marked
    attrs: srs_q, srs_kind          what an SRS was worked out at
           interpolation           how a spec reads between its lines
           sine_tone, sine_onset   which tone a sine level answers to,
                                   and where its sweep began; the
                                   sine_seconds dataset is its clock
           scale_db                a hand-held comparison scale only —
                                   auto-detected scaling is never saved
           averaging_frame_length, _overlap, _window, _frames, _start,
           _detrend, and _window_parameter when the window takes one
           filtering_low, _high, _order   the filter reading, when set
           truncation_start, _stop        the truncate reading, when set

## `shapes/`

    attrs   mass_unit ('' = undeclared), unscaled (no drive point —
            modal masses are a convention, not physics)
    dsets   frequency, damping, shape_matrix (modes × DOFs, complex
            where a source carried complex modes), modal_mass
            (complex where measured), coordinate, comment, description
            modal_damping        only where a source carried one

## `channel_table/`

    attrs   column_order         every column, in the file's order
    dsets   channel, range       integers — the schema's 'int' columns
            <every other column> strings, verbatim (node included: a
                                 node nobody recorded is blank, and 0
                                 cannot stand in for 'unstated')

## `report/`, `photos/`, `matches/`

    report/   attrs: title, marking, marking_color ('ink' | 'red';
              marking '' = no strips);  dsets: blocks (one JSON string)
    photos/   dsets: names, formats, image_0000... — the photograph
              files' own bytes, verbatim, never re-encoded
    matches/  attrs: first, second, first_geometry, second_geometry
              dsets: pairs (n × 2 mode indices), macs

## `sine_specification/`, `sine_levels/`

    sine_specification/  attrs: ordinate_dim, ordinate_unit, comment
                         dsets: response_dof, tone_order
                         tones/<k>/: attrs name, start_time; dsets
                         frequency, amplitude, phase, segment_type,
                         segment_rate, and any warning/abort limits —
                         the file's own breakpoint shape, per tone
    sine_levels/         dsets: tone_order; tones/<k>/ each an
                         ordinary data/ layout — the set is grouping,
                         not a new format

## Reading one without Visual Dynamics

The point of the layout being documented HDF5 is that this works:

```python
import h5py

with h5py.File('modal.vdyn') as f:
    for key in sorted(f['objects']):
        group = f['objects'][key]
        print(group.attrs['name'], '—', group.attrs['kind'])
    frf = f['objects/0003']                    # say it's the FRF
    H = frf['ordinate'][()]                    # records × frequencies
    freq = frf['abscissa'][()]
```

MATLAB reads the same file with `h5info('modal.vdyn')` and
`h5read`. Nothing in a `.vdyn` needs this package to get back out.
