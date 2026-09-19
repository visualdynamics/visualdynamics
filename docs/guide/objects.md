# The objects

A project holds objects, and every object is a plain Python thing
with arrays in it. The window is a view onto them; a script holds the
same objects and calls the same methods. This page is the map: what
kinds there are, what each one stores, where it comes from, and what
can be done with it. The API reference has the signatures — each
heading below links to its page.

Two rules hold across all of them:

- **Values are stored in SI** once their units are known. A record
  whose units were never declared keeps the file's raw numbers and
  says so (`ordinate_dim == 'unknown'`); nothing is ever scaled by a
  guess. [Units](units.md) has the whole story.
- **A project reaches an object by its kind** — `project.frf`,
  `project.time_histories` — as [Projects](projects.md) explains, and
  `project.verbs(obj)` lists the processing verbs that apply to any
  object, each with its one-line reading, and
  `project.selection_verbs(*names)` the ones a selection of several
  can take together. Those lists and the acts on the window's bar
  read one table, so they cannot disagree.

## Geometry — [`core.geometry`](../api/visualdynamics.core.geometry.md)

Kind `geometry`. Nodes, coordinate systems, tracelines, elements and
blocks, held as flat arrays: `node_id`, `node_xyz` (meters), the
placement and measurement system of each node, `cs_matrix` (three
direction rows and an origin per system), one connectivity array per
traceline and per element, and the block each element belongs to with
its name. `nodes`, `coordinate_systems`, `tracelines`, `elements` and
`blocks` are row **views** onto those arrays — what the tree lists and
what a script usually reaches; writing through a row writes the
array.

Comes from universal files, Exodus, Nastran and Femap decks, STEP and
IGES, 3MF and STL meshes, sdynpy arrays and `.vdyn` files. Does:
`plot` and `plot_dofs`; `add_node`, `add_traceline`, `add_element`,
`add_block` and their `delete_*` and `renumber_*` counterparts;
`missing_dofs` against a data object; `extent`; `validate`; `save`.

A geometry also carries `mass_properties` — the reference point its
rigid-body modes pivot on and, when the set is to be mass-normalized,
the mass and inertia tensor about it
([`core.rigid.MassProperties`](../api/visualdynamics.core.rigid.md)).
They ride the geometry the way averaging rides a time history: set in
the rigid-body view (the toggle on the 3-D view's bar, offered to one
selected geometry), saved with it, and `suggest_mass_properties` seeds
the centroid with no mass. *Generate Rigid Body Mode Shapes*
(`project.generate_rigid_body_modes`) makes the six-mode shape set in
the geometry's group, previewed on the model as the point is set.

## Photos — [`core.photos`](../api/visualdynamics.core.photos.md)

Kind `photos`. Setup photographs as they arrived — `names`, `formats`
and the encoded `images` — never re-encoded, because a report embeds
them and a second JPEG generation gains nothing. Does: `add_file`,
`rename`, `delete_photos`, `plot`.

## Channel table — [`core.channel_table`](../api/visualdynamics.core.channel_table.md)

Kind `channel_table`. One `frame`, a row per channel and a fixed,
typed column set. Comes from a controller run or a spreadsheet. Does:
`controls`, `sensitivities`, `ranges`, `dof_strings`, `units_for`,
`set_cell`, `rename_dof`, `delete_channels`, `save`. [The channel
table](channel-table.md) has its rules.

## The data arrays — [`core.data`](../api/visualdynamics.core.data.md)

Every measured or computed curve is a `DataArray` subclass, and one
object holds **many records** sharing one abscissa: `abscissa` (the x
axis — seconds or hertz), `ordinate` shaped `(records, samples)`, and
one entry per record of `response_dof`, `reference_dof` where the
type has a reference, `block` (which repeat), `ordinate_dim` and
`ordinate_unit` (what and in which SI unit), `reference_unit` for a
ratio, `dimension_hint` (what a file claimed without saying its
scale) and `comment`. Uneven and unsorted abscissas are allowed at
the door; anything that needs an even step asks at the point of use.

What every data array does: `plot`, `plot_waterfall`, `save_plot`,
`save`; `define_units` and `undefine_units`; `delete_records` and
`rename_dof`; `display_ordinate` and `display_abscissa` in a chosen
unit system; `num_records` and `record_label`.

The rows of a data object's grid in the project tree are its
coordinates, and the columns of a matrix are coordinates too. A
channel assigned to the wrong point at the instrument is corrected
there: double-click the row or column label, type the coordinate,
and the channel takes it — that channel, the coordinate and the
quantity the row or column is, wherever it appears in the object: a
CPSD's accelerometer is on both sides of its cross terms and moves as
one sensor, while an FRF's drive-point accelerometer (a row) and load
cell (a column) are two channels, and moving one leaves the other to
be moved explicitly. Everything derived from the object follows,
because a PSD computed from a mislabeled channel is mislabeled the
same way (`project.rename_dof`; without a quantity it moves every
channel at the point). A rename that would give two records one
coordinate and one quantity is refused: a load cell and an
accelerometer share a point, two accelerometers do not. A channel
table's row coordinate *is* its node and direction: type over the
row in the grid and the two cells change, type a node or a direction
in the table view and the row's coordinate follows. A channel table
imported beside the data it describes is linked with it — one file,
one group — and is otherwise its own object: a coordinate corrected
on the time history is not corrected on the table, or the reverse.

| kind | class | what it is | what it adds |
| --- | --- | --- | --- |
| `time_history` | `TimeHistory` | the record as acquired, against time; carries its `averaging` and `shocks` readings so every derivation reads one description | `compute_spectra`, `compute_psds`, `compute_cpsds`, `compute_frfs`, `compute_multiple_coherence`, `compute_srs`, `integrate`, `differentiate`, `filter`, `truncate`, `split_into_frames`, `sample_rate`, `drive_dofs` |
| `transient_specification` | `TransientSpecification` | a target waveform: what a transient test was controlled to, sample by sample | everything a time history does |
| `spectrum` | `Spectrum` | the complex average of a record's frames — amplitude and phase per line | `animate` |
| `psd` | `Psd` | the power average: real autospectra, complex cross spectra; `bin_widths` when the bins are not even | `area`, `to_octave`, `bin_bounds`, `principal_shapes`, `animate` |
| `specification` | `Specification` | a PSD with its band: the target and the warn and abort limits either side, as arrays beside the ordinate | `limit`, `has_limits` |
| `frf` | `Frf` | response per unit reference, complex, one reference per record | `plot_cmif`, `animate`; the input to a modal fit |
| `coherence` | `Coherence`, `MultipleCoherence` | how much of a response one reference explains, or all of them together; bounded 0 to 1 | `plot_map` |
| `srs` | `Srs` | a shock response spectrum: the peak an oscillator of each natural frequency reached, laid out in octaves | `damping` |
| `shock_specification` | `ShockSpecification` | an SRS with its band, conventionally +6 dB and −3 dB | `limit`, `has_limits` |

## Sine — [`core.sine`](../api/visualdynamics.core.sine.md)

Two kinds that are not data arrays, because each tone sweeps its own
frequencies on its own clock and different abscissas cannot share
one. `sine_sweep_specification` (`SineSweepSpecification`) is what a
sine test was controlled to: the `tones`, each with its breakpoints,
sweep law, bands and start time, over the control DOFs. `sine_levels`
(`SineLevelSet`) is one extraction — the per-tone `levels` a joint
Vold-Kalman solve read out of a recording, grouped the way the
specification groups its tones.

## Shapes — [`core.shapes`](../api/visualdynamics.core.shapes.md)

Kind `shapes`. Mode shapes over a shared set of DOFs: `frequency` and
`damping` per mode, `shape_matrix` shaped `(modes, dofs)` with
`coordinate` naming each column, `modal_mass`, `modal_damping` and
`mass_unit` where a source carried them, a `description` per mode, and
`unscaled` when the fit had no drive point to pin the scale. A fitted
set is also the record of its fit, which is what *Edit Fit* reopens.
A geometry's rigid-body set (three translations, three rotations
about its reference point, `frequency` exactly zero) is `unscaled`
too unless mass and inertia were given, in which case it is
mass-normalized about the inertia's principal axes.

Any set carries data through itself: `project.transform` fits a
record's motions to the modes (`q = Φ⁺u`) and projects its forces
(`Φᵀf`), one record per mode and quantity at the modal coordinates
`M1` … `Mn` — the letter and the mode's index, visibly not a node, the
same for every set; the object's provenance says which set — and
`project.expand` carries modal responses back to every DOF the set
covers. Every kind of data goes the same two ways in the form its
kind takes: a time history or spectrum as rows; a CPSD or a
specification as the matrix, `S_qq = Φ⁺ S_uu Φ⁺ᴴ`, which needs every
cross term between the shared channels — autospectra alone are refused
rather than completed with a guess, so compute the CPSDs, or import
the specification with the cross terms the controller wrote — and its
tolerance bands carried exactly when every channel wears the same one;
an FRF on its response rows and, when the set covers the drives, its
reference columns too. A shock response spectrum or a coherence does
not transform (a maximum, a ratio): transform the time history and
compute it again. A specification can also be *written*, on a
sheet that opens three ways: *Specification* on the table bar of a
lone shape set starts one at its modal coordinates — every shape, or
the shapes picked in the tree, so a virtual point's target is its
three translations and the rotations, left out, contribute nothing
when it is expanded; the same button
on a lone channel table starts one at its control channels; *Edit*
on the plot bar of a specification opens the specification itself
— all of its channels, or the ones whose records are picked in the
tree, and several specifications at once open as one sheet, a
column per channel under its specification's name, cross terms
within each. The sheet sits beside the plot — breakpoints and a
level per channel, every pair's coherence and phase (a pair left
unstated is absent; independent is a statement too), the a box that scales the selected levels by decibels. While the sheet is open the plot is the flat
one — every editing gesture lives there, so the 3-D stage stands
down until the sheet closes. The warning and abort bands are not on
the sheet at all: they are on the plot, as the shaded zones, and while the sheet is open each edge of each band
carries a drag handle: one over the whole range while the channel is
*Uniform*, and with Uniform off one per linear section of the
requirement — every segment between breakpoints, or every run of one
power law in the interpolated form — each moved on its own. Drag
one up or down — it snaps to quarter decibels, and its label says
which — and, when you let go, that edge moves to that value on every
channel the sheet holds in that frequency range: the whole
specification, or the channels picked in the tree. A specification
with no limits opens wearing the defaults, ±3 dB warning and ±6 dB
abort, there on the plot to drag; the sheet says they are not on the
object yet, and the first edit writes them. Two constraints
sit in the sheet's head beside the form buttons and apply to every
channel of the specification the same way: *Symmetric* keeps the
band above at minus the band below, *Uniform* keeps every segment at
one band. Both are on to begin with, and a specification opens with
whatever its own bands say. Both grids are the application's ordinary tables: a column
header selects a channel, a row header a breakpoint, Cmd-click adds
cells, and copy, paste, Delete to clear, Batch Edit and the fill
handle work as they do everywhere else; a frequency typed between
two others re-sorts the breakpoints. There is no button: every
edit lands on the specification as it is made, so switching to
another object never leaves an edit behind. Opened on a shape set
or a channel table, the sheet first makes the specification beside
it and edits that. A sheet holding every channel rewrites the
object in the sheet's own form — fold a controller's target to its
breakpoints and the object *is* the breakpoints; a sheet holding a
picked subset of channels merges back at the object's own lines
with the other channels untouched, so the name, links and place in
every report hold (`project.author_specification` with a
[`core.author.SpecificationDraft`](../api/visualdynamics.core.author.md);
`replace=True`, and a list of names for a sheet spanning several).
A pair left unstated is absent from the specification, not
assumed; a cross term that varies with frequency opens unstated,
and the sheet says so. A sheet has two forms, chosen by the pair of buttons
at its top and applied to the whole specification whichever
channels the sheet is open on: *Breakpoints*, the few points a requirement is written
from, and *Interpolated*, the same requirement read onto evenly
spaced frequency lines as a controller writes its target. A
controller's target opens as lines and folds to its breakpoints
exactly, since its lines were read from them, and editing is done on
the few rows; a specification with no power-law structure keeps every
line and says so. The spacing the lines are read onto is the
specification's own, else a time history's averaging (sample rate
over frame length), else typed into the box beside the buttons.
Nothing is guessed. Written at modal coordinates and expanded through
the set, it is an exact control-channel target with every cross
term. *Save As* offers a specification as *Rattlesnake random
specification (.npz)* — the target file the controller's Random
environment loads before a test: the frequency lines, the whole
cross-spectral matrix at each, the four bands, and the node and
direction of every channel so the controller puts the matrix in its
own channel order. Most random tests run on autospectra alone, and
the controller's form of that is a matrix with zeros off the
diagonal, so a pair the specification does not hold is written as a
zero and a `held` mask records that it was a placeholder rather
than a statement of independence. The values go out in the
coherent unit system on display, which the file cannot record and
the status line names. The same file imports again, held cross
terms kept and placeholders left out; a controller's or sdynpy's
own target file imports too, an off-diagonal all zero or NaN read
as its placeholder for autospectra alone. The expansion
goes back — every mode for the motion of the structure, or the modes
picked in the tree for their contribution alone, `u = φₖqₖ`, the
result named for them (`records=` in a script). Both land in the
group with the set and the record — a modal object answers to the
shape set it came through rather than to a geometry, and fits where a
set has every mode it names — and both carry the record's averaging
frames and shock windows, which land on the same instants. The unit rule is
`[q] = [u]/[Φ]`:
unit rigid shapes give the virtual point's rotations in rad/s² and its
moments in lbf·in, mass-normalized shapes give the half-power mass
units (`modal_acceleration`, in/s²·slinch½), and a set with no mass
unit gives responses to declare. In the window, select the record and
the set together and press *Transform to Modal Responses* on the
bar (or *Expand to Physical Responses*, for a modal record); it
makes the object at once — a transform has no settings — and the
status line gives the account: DOFs shared, left out, the residual
the fit leaves, kept on the object as its `transform_report`
([`core.transform`](../api/visualdynamics.core.transform.md)).
Does: `plot`, `animate`, `plot_mac`, `auto_mac`, `synthesize_frf`,
`covers`, `delete_modes`, `save`.

## Matched modes — [`core.matches`](../api/visualdynamics.core.matches.md)

Kind `matches`. Pairs committed on a cross-MAC between two shape
sets: `first` and `second` name the sets, `pairs` the modes, `macs`
the value each pair had when it was committed. Does: `add`,
`delete_matches`.

## Report — [`core.report`](../api/visualdynamics.core.report.md)

Kind `report`. An ordered list of blocks, each a plain dict bound to
project objects symbolically by default (`@basis:Frf`) or by a literal
name: text, plot, scene, table, photo, bars, the
matched-modes pairs and their overlay. Does: `add`, `remove`, `move`,
`figures`, `unbound`. [Reports](reports.md) explains the bindings.

## What can be done with what

The processing verbs live on the project, so a result is added,
named and linked to its source in one call:

| on | verbs |
| --- | --- |
| a time history | `filter_data`, `truncate_data`, `detect_shocks`, `compute_spectra`, `compute_psds`, `compute_cpsds`, `compute_srs`; `compute_frfs` and `compute_multiple_coherence` when it has drive channels; `extract_sine` when the project holds a sine sweep specification; `integrate` and `differentiate` when the quantity allows; `transform` through a shape set whose DOFs it is measured on, and `expand` back when it holds that set's modal responses |
| a geometry | `generate_rigid_body_modes` |
| a PSD, CPSD or specification | `compute_octave` — a specification's warning and abort limits band with it |
| an FRF | `fit_modes` |
| two shape sets | `project_onto_basis`, `match_modes` |
| any object with a sibling of its type | `merge` |

Every object also answers to the whole-project verbs — `add`,
`remove`, `rename`, `duplicate`, `link`, `unlink`, `place`,
`set_basis`, `export`, `save`, `generate_report`, `export_report`,
`refresh` — and `project.verbs()` with no argument lists all of the
processing verbs with their readings. The API page for
[`project`](../api/visualdynamics.project.md) has every signature.
