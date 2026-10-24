# Importing and exporting

Visual Dynamics can write what it reads as test data, so the data can
go on to whoever needs it next. The formats marked read-only below are
the exceptions: solver results (a Nastran punch file, a Femap neutral
file) and controller runs (a Rattlesnake `.nc4`) are produced by other
programs, and a file written here would claim to come from a program
that never made it.

The native format is `.vdyn` (HDF5), and it is the only one that keeps
everything a project holds, units included. Saving and opening a
project use it. While Visual Dynamics is an alpha its format may still
change ([the `.vdyn` page](vdyn-format.md) says what that means for a
file you have). It is standard HDF5 with a documented layout, given on
that same page, so any HDF5 tool reads one without this package. The
other formats are for exchanging data with other tools, and each one
loses whatever it has no way to record.

One row per format; R/W means both read and written:

| format | ext | Geometry | ShapeSet | function data | ChannelTable |
|--------|-----|----------|----------|---------------|--------------|
| Universal file ⁴ | `.unv` (read also as `.uff` and `.uf`) | R/W | R/W | R/W | — |
| I-DEAS ADF ² | `.afu`/`.ati`/`.ash` | — | R/W | R/W | — |
| Exodus ¹ | `.exo` (read also as `.e`, `.exo2`, `.g` and `.gen`) | R/W | R/W | R/W | — |
| Nastran bulk ³ | `.bdf`/`.dat`/`.nas` (written as `.bdf`) | R/W | — | — | — |
| Nastran punch ³ | `.pch` | — | R | — | — |
| Femap Neutral ³ | `.neu` | R | R | — | — |
| sdynpy | `.npz` (geometry, data), `.npy` (shapes) | R/W | R/W | R/W | — |
| Rattlesnake run ([below](#rattlesnake-runs)) | `.nc4` (read also as `.nc`) | — | — | R | R |
| Rattlesnake random specification | `.npz` | — | — | R/W | — |
| Rattlesnake sine specification | `.npz` | — | — | R | — |
| Excel | `.xlsx` (read also as `.xlsm`) | — | — | — | R/W |
| 3MF CAD mesh ⁵ | `.3mf` | R/W | — | — | — |
| STL mesh ⁵ | `.stl` | R/W | — | — | — |
| STEP / IGES ⁶ | `.step`/`.stp`/`.iges`/`.igs` | R | — | — | — |
| MATLAB ⁷ | `.mat` | R/W | R/W | R/W | R/W |
| Engineering Sciences Common Data Format ⁸ | `.h5` | R/W | R/W | R/W | R/W |
| Delimited text ⁹ | `.csv` (read also as `.tsv`, `.txt`, `.tab`, `.asc` and `.prn`) | — | — | R/W (time histories) | — |

Three more go in and out whole rather than as one object. The first is
the project file itself: `.vdyn`, with everything a project holds;
`.mat`, the same layout as MATLAB structs (note ⁷); or `.h5`, the
standard's types with each object whole beside them (note ⁸). The
second is a report template (`.vdreport`), a report on its own to load
into another project (see [reports](guide/reports.md)). The third is
photographs, a folder of images (`.png`/`.jpg`/`.heic`), through the
project's Photos.

"Function data" uses the dataset 58 vocabulary throughout: time
histories, FRFs, PSDs, spectra, coherence, multiple coherence and SRS.
Not every format carries every kind. Exodus takes time histories and
spectra on a mesh, and ADF splits functions and time data between
`.afu` and `.ati`.

A channel table is the one object that is a spreadsheet, so it gets a
spreadsheet format: every column in order, channel numbers as
integers, the rest as text. Its units are metadata, not values, so
nothing is converted.

## Notes on the formats

¹ Mode shapes go to exodus on a geometry. The file is a mesh with
results, one time step per mode with the frequency as the step's time,
which is what ParaView animates (its reader assembles DispX/Y/Z into a
`Disp` vector for Warp By Vector). Shapes without a geometry are
refused, because a modal exodus file with no mesh is a file nothing
can draw. In the app, *Save As…* puts the shapes on the geometry linked
to them, otherwise on the active geometry (set with *Set as Active
Geometry*), otherwise on the project's only geometry. The status line
names the geometry used, and a geometry holding none of the shape DOFs
is refused. From a script, pass `geometry=` to `export_file` or
`exodus.save`.

Damping and modal mass go out as two global variables, `Damping` and
`ModalMass`, one value per step. A complex shape goes out as its real
part under the plain name, so ParaView still assembles its `Disp`
vector, with the imaginary part beside it as `DispX_IM` and so on, and
the reader puts the two back together. DOFs at nodes the mesh does not
have are dropped.

The geometry's coordinate systems go out as the file's coordinate
frames (an id, three points and a rectangular, cylindrical or
spherical tag) and come back as systems, with their definitions but no
names. Nothing in exodus lets a node or a variable refer to a frame,
so shape values at a node with a local displacement system are rotated
to global on the way out. Which node is placed or measured in which
system is written as named node sets (`def_cs_7`, `disp_cs_7`), a
standard feature that other exodus tools show as ordinary node sets
and that this reader restores the assignments from. A modal model
makes the round trip through Visual Dynamics without loss; another
tool reads the mesh, the modes and the frames, and sees the node sets
as node sets.

Time histories and spectra go on a geometry the same way. The abscissa
becomes the file's step axis, and each (quantity, direction) becomes a
nodal variable named so that it reads back as itself (`AccX`,
`Pressure`). A complex spectrum is written as `_RE`/`_IM` pairs, and a
record with no DOF becomes a global variable named by its block label
or its comment. A (node, variable) with no record is NaN, never zero,
because a missing reading is not a reading of zero. Values in a node's
local displacement frame are rotated to the global frame, as shapes
are. A stacked object (averages) is refused with the collision named,
since one variable per node cannot hold twenty readings of one
channel; select records first.

To read such a file back you say what its steps are, with
`load(steps='time')` or `load(steps='frequency')`, or in the app's
*Reading the results* question, because the file cannot say what its
own step axis means. The default reading is one mode per step. Exodus
records no units in either direction: values go out in the display
unit system and come back raw, with the variable's stem kept as a
quantity hint. A mode whose step time is negative (an eigensolver's
rigid-body mode, whose eigenvalue comes out slightly either side of
zero) is read as 0 Hz, the file's own number is kept in the mode's
description, and the import says so.

² ADF is the I-DEAS Associated Data File format, which Siemens NX also
reads. Files written here were checked by reading them back with the
reference tooling, value for value, including complex shapes and
complex modal mass and damping. Two properties belong to the format
and no writer can change them. ADF stores SI whatever the display
system (the one exception to the display-units rule below; the format
defines it as an SI-writing dataset 164 in binary), and ADF stores
float32 only, so double-precision data is narrowed on the way out. A
shape file fills each node out to whole X/Y/Z (or 6-DOF) vectors, with
zeros at directions the set does not carry, and the importer's
zero-column rule drops those again on the way back in, so the round
trip is stable. The layout the readers and writers follow was derived
from sample files and is held to them by tests.

The same three extensions import like any other file. Since an ADF is
SI by definition, its data arrives with units declared, not hinted.
The exception is a record typed Unknown or General: it was never
converted to SI, its values are raw, and its ordinate units label
(free text, and often wrong in real files) arrives as a hint only,
naming the axis and narrowing the shortlist in the *Imported Units*
pane. You declare the units yourself, because nothing is scaled on a
guess.

³ These formats carry a finite element model and its modes, so that a
test can be set beside a model:

- Nastran bulk data is read and written as a mesh. The reader
  takes grids (all three field formats, bare-exponent floats
  included), CORD2R/C/S coordinate systems with chained references
  resolved, elements, concentrated masses (CONM2), and PLOTELs as
  drawn lines. PSOLID, PSHELL and MAT1 cards become the element
  groups' materials and thicknesses, with the deck taken to be in SI.
  CELAS1 and CELAS2 springs are read with their stiffness (a CELAS1's
  from its PELAS), one element group per stiffness and direction, and
  SPC and SPC1 cards become ground points held in their components. A
  deck with more than one SPC set, an SPC that enforces a
  displacement, a spring joining two different components, a CORD1
  system, or a connection card the reader does not know is refused by
  name. Rigid elements (RBE2, RBE3, RBAR, RROD), MPCs, loads and
  control cards are skipped. The written deck is for interchange, not
  a runnable model: grids go out as large-field GRID* cards at full
  precision, elements carry PID 1 with no property or material cards
  (Nastran would need PSHELL/PSOLID cards added before it runs),
  springs go out as CELAS2 with their stiffness, point masses as CONM2,
  ground as SPC1 and drawn lines as PLOTEL chains, and the header line
  says Visual Dynamics wrote it. Coordinates are written in SI, as the
  geometry stores them, whatever the display system.
- A Nastran punch file brings the modes: SOL 103 eigenvectors read
  into a ShapeSet, with the frequencies from the eigenvalues. Complex
  output is refused by name.
- A Femap neutral file reads both halves from one file, the geometry
  (blocks 403 and 404) and normal-modes output (450 with 451 or 1051),
  against layouts taken from real files. A version whose record layout
  differs is refused, naming the version, rather than misread. Rigid
  elements and nodes defined in local coordinate systems are refused
  by name; static and stress vectors are skipped.

Modes imported from a punch or a neutral file arrive unscaled. Neither
format records its normalization, so shapes and MACs are exact, and
nothing downstream reads a physical scale out of modal masses that
were never in the file. The docstrings of `io/nastran.py`,
`io/punch.py` and `io/femap.py` list every rule each reader follows.
OP2 is planned and waits on a real file to test against. Abaqus ODB is
not read: it is a proprietary binary whose only practical reader is
Abaqus's own API. The BK Connect and Vibrata formats are not read
either, since they are proprietary, with no public specification and
no sample file to work from. Any of the three would need a real file
and a real need.

⁴ The universal file has two encodings, and both are read. An ordinary
dataset 58 writes its values as `13.5E` text; dataset 58b keeps the
same eleven ASCII header records and then writes the values as raw
IEEE 754 floats. The binary form is roughly a third smaller and, unlike
the text form, exact: five decimal places do not survive a round trip,
and raw doubles do. LMS Test.Lab and I-DEAS both write it.

Reading takes either form, in either byte order, at either precision,
and tells the two apart by the `b` on the dataset's own marker line.
Only IEEE 754 floats are decoded. Dataset 58b can also declare DEC VMS
or IBM 5/370 floating point, which are different representations, not
different byte orders, and reading one as IEEE would return plausible
wrong numbers instead of failing, so the reader names the format it
found and stops. The specification is mirrored at
`docs/uff_spec/58b.asc`.

Writing uses the text form by default, because every reader takes it.
The binary form, written as little-endian doubles, is
`visualdynamics.io.unv.save(obj, path, binary=True)`, or the
`unv_binary` exporter (`export_file(obj, path, format='unv_binary')`),
which is also its own entry in the app's *Save As…* dialog. Only the
functions change form; geometry, mode shapes and the units dataset have
no binary variant.

⁵ 3MF and STL are for CAD that is already tessellated: export either
one from SolidWorks or any CAD tool. 3MF keeps each element group as a
named part and declares its unit (it is written in meters). STL has no
units and, in its usual binary form, no part names, so a binary STL
arrives as one element group named for the file; the rare ASCII form
gives an element group per `solid`. Both hold faces only, and a
geometry without face elements is refused. On mesh density: this
geometry is context drawn behind the sensor points, so coarse export
settings in the CAD program are enough, and a fine tessellation costs
drawing time without adding anything.

⁶ STEP and IGES are B-rep CAD, tessellated on import by the bundled
OpenCASCADE kernel (LGPL-2.1 with exception, one more replaceable
library; see NOTICE.md). Named parts become named element groups, and
coordinates are converted to meters from whatever the file declares.
These formats are import only: Visual Dynamics holds meshes, not
surfaces, so a B-rep written from one would be made up. A pip install
keeps the kernel optional (`pip install 'visualdynamics[step]'`); the
packaged application includes it.

⁷ MATLAB `.mat` is the project file in MATLAB's container, not a
foreign format. A `.mat` written here holds what a `.vdyn` holds, under
the same names ([the `.vdyn` layout](vdyn-format.md)), so every kind of
object, the project itself included, goes out and comes back.
`load('modal.mat')` gives `objects`, a cell array of structs, each with
its `name` and `kind`, alongside the project's `test_name`,
`project_type`, `links` and `provenance`. A single object saved alone
is one struct named for its kind (`data`, `geometry`, `shapes`, …). The
changes MATLAB needs are mechanical: ragged connectivity is a cell
array of row vectors, one-dimensional datasets are column vectors,
strings are char arrays and lists of them cellstr, and `''` still means
undeclared.

Values are in the display units, and every unit field names the unit
its values are in. With in-slinch-lbf-s (g) chosen, an acceleration is
in g with an `ordinate_unit` of `'g'`, an FRF in in/s² per lbf with
`'in/s**2'` and `'lbf'`, and a geometry's coordinates in inches with a
`length_unit` of `'in'`. A `unit_system` variable names the system,
and reading the file back takes every value to SI from its own unit,
so a record comes back declared in the unit it was written in.
Material and section properties stay in SI, as the project file holds
them, and an abscissa is in seconds or hertz in every system. Saving
the project as `.mat` (*Save Project As…*, or `project.save` in a
script), or exporting from a script with no `unit_system`, writes the
project file exactly: SI values, each record beside the unit it was
declared in. A file with no `unit_system` variable is read as SI.

The reader takes back the files it writes, and one other thing: a
single struct built by hand to the documented layout of one object,
without the schema stamp. Anything else in a `.mat` is refused by name
rather than guessed at. Only version 5 files are read and written
(MATLAB's `-v7`), which hold at most 4 GB per variable. A larger record
is refused before anything is written, and the `.vdyn`, which MATLAB's
`h5read` opens directly, is the file to hand over instead.

⁸ The Engineering Sciences Common Data Format (ESCDF) is a public
standard built on HDF5. A geometry, a channel table, curves and modes
are each a type defined by a plain-text specification file, and
activities hold results and link the metadata that defines them. A
project written here is one file. Every geometry and channel table is
a metadata group, and every curve set and shape set is a result in the
activity of its object group (a named group is an activity of that
name). Each object is also stored whole, in its own `.vdyn` layout, as
an attachment of its group, so the file reads back here exactly, and
any other reader of the format gets the standard fields with nothing
of ours in the way. A report is attached twice, rendered as a complete
HTML page and as its definition, so it reopens editable here, and
photographs are attached as the images they are.

A file another program wrote is read by its standard fields: nodes,
lines, elements and per-node axes (a node with its own axes gets a
coordinate system), records with their units, frequencies and shapes,
and activities as named object groups. A type the specifications do
not define is left out with a note. The reader and writer were checked
against the format's reference implementation in both directions, and
the specification files that define the types ship inside the package.
A file is written as `.h5`, the container's suffix, which is how the
format's files are named where it is used; a file named `.hdf5` or
`.escdf` keeps its name, and reading looks at the contents, never the
suffix.

⁹ Delimited text is the column export of controllers, recorders and
spreadsheets: one column per channel, with or without a time column,
a header and a preamble. The reader finds the delimiter, the decimal
point, the header, the units, the time column and a stated sample
rate, takes any of them as stated instead, and refuses with the line
and column what it cannot read; [Importing a time history from
delimited text](guide/text-import.md) gives the rules. A time history
is written back as a `.csv` with a header of names and units that the
reader takes back in, in the display system itself, g included.

SEP 005, the sdypy project's standard for time series, is exchanged in
the API rather than as a file, since it is an in-memory dict format.
`visualdynamics.from_sep005(...)` reads the dicts a sdypy package hands
over, `TimeHistory.to_sep005(...)` produces them, and the exported form
passes sdypy's own compliance validator.

`export_file(obj, path)` picks the exporter by suffix, and `format=`
names one outright. Where two exporters share a suffix the first one
listed is taken: `.unv` gives the text form, and `.npz` gives sdynpy
data, so a specification written for Rattlesnake needs
`format='rattlesnake_specification'`. In the app, *Save As…* on an
object offers `.vdyn` first and then only the formats that object can
be written as, so a mode shape is never offered a format that cannot
hold one. Exodus is offered for shapes and data, and the save is
refused in the status line when the project has no geometry for them
to go on (note ¹).

## Units

Exports are written in the display unit system. What is on screen is
what goes in the file: with in-slinch-lbf-s selected, a model one inch
across is written as `1.0`, not as `0.0254`. A display system that
shows acceleration in g (such as in-slinch-lbf-s (g)) writes the
coherent system it is built on, in/s² in that case, because MATLAB's
`.mat` is the only one of these formats that labels a value as g; the
status line names the system written. The exceptions are ADF, whose
definition requires SI on disk (every ADF reader converts on its own
side, so the numbers still arrive right); Nastran bulk data, written in
SI; 3MF, written in meters and declared as meters; ESCDF, where each
value is written in the unit it was declared in and labeled with that
unit (a geometry in inches as inches, a record in g as g); and MATLAB,
written in the display units including g, with every unit field
labeled (note ⁷).

UNV says which units it used, in a dataset 164 declaring the system and
its conversion factors, so the file comes back to exactly the same SI
values whatever system it was written in. 3MF, MATLAB and ESCDF also
carry their units. sdynpy, exodus, Nastran, STL and the Rattlesnake
specification record none, so they carry the numbers you were looking
at, and the status line after the save names the units, since it is
the only place they are stated.

An object with no units declared is written exactly as it stands.
There is no factor to convert it by, and multiplying by one would claim
it was SI. No dataset 164 is written, and it re-imports unit-less,
which leaves the next reader in the same position.

A dataset 58 can name a quantity without giving its size. A record
that knows it holds accelerations, because the file it came from said
so (kept in `dimension_hint`), goes out as data type 12 with no 164 to
scale it, and reads back knowing the same thing. Only a record that
knows nothing is written as data type 0.

In a script, `export_file` without a `unit_system` writes the stored
values, which are SI once units are declared.

## What each format drops

sdynpy keeps geometry, including coordinate systems, drawn lines
(its tracelines) and elements; data, including reference DOFs; and
shapes, including frequency, damping, modal mass and the DOF set. Where
Visual Dynamics has both a `comment` (from the file an object came
from) and a `description` (typed in Visual Dynamics), the format has
one field, and the description is written.

Mode shapes go to UNV as one dataset 55 per mode, the standard
nodal mode-shape record. Real modes are written as normal modes
(analysis type 2), with the frequency, modal mass and damping ratio
and the vector at each node. Complex modes are written as complex
eigenvalues (analysis type 3), with the eigenvalue, modal A and modal
B, which is where curve fitters put them and where readers look for
them. Both types are read back. A complex set's modal A is what
`ShapeSet.synthesize_frf` scales it by. Exodus stores modes as it
stores any nodal result: one time step per mode, with the mode's
frequency as the step's time, and nodal variables `DispX`/`DispY`/`DispZ`
(plus `RotX`/`RotY`/`RotZ` when the shape carries rotations). Damping
and modal mass are the file's global variables `Damping` and
`ModalMass`, one value per step, so a modal file comes back with
everything it went out with.

Dataset 55 stores a whole vector per node, so a shape covering only
some directions is padded with zeros and comes back covering all of
them.

### UNV keeps local frames; exodus keeps them beside the values

The two formats can say different things, so they are treated
differently.

Dataset 55 takes a node's values in
that node's own displacement system, so the values are written as they
stand, and a dataset 2420 is written alongside defining every
coordinate system: label, type, name and its 4x3 transform. The node
references (`node_def_cs`, `node_disp_cs`) come from dataset 2411. A
file exported this way makes the round trip with its frames intact.

*Caveat:* the published specification for dataset 2420 could not be
found. The SDRL index lists it, but its file is not on the site,
although the datasets mirrored under `docs/uff_spec/` are. The layout
used here makes the round trip through Visual Dynamics but has not been
checked against the standard or against a file from another tool. If
you have the specification, add it to `docs/uff_spec/` so it can be
checked.

### Exodus results are written in the global frame

Exodus can hold coordinate frames, and the geometry's systems go out
as them, but nothing in the format lets a node or a variable refer to
one. A shape value left in a node's own displacement frame would be
read back as though it were global, and be wrong without any warning.
So values are rotated into global on the way out, translations as one
triple and rotations as another, with the same convention the animator
uses. That needs the geometry, since the shapes alone do not say what
frame they are in: pass `geometry=` to `exodus.save`, which also writes
the mesh alongside. Without it the values go out as they stand. Which
node was placed or measured in which frame is not lost: it is written
as the `def_cs_<id>` / `disp_cs_<id>` node sets described in note ¹,
and the reader restores the assignments from them, so the systems, the
assignments and the global values together reproduce the model
exactly.

### Element groups survive the round trip

An exodus mesh is written as named element blocks, and the block is
where a part's identity is kept: `wing`, `tail`, `nacelle front left`.
A geometry carries them as element groups (`elem_group` per element,
with `group_id`/`group_name` declaring them), so a file read and
written back keeps its groups.

They matter beyond the round trip. `fem.Model.from_geometry` reads the
section of a member from the element group of the element it came
from, so a saved geometry and `sections={'prop': BLADE}` are a complete
description of a structure, and a geometry whose element groups carry
their own properties builds with nothing passed. Asking a node instead
would not work: a node on a seam between two element groups belongs to
both and can answer for only one.

Formats that do not record element groups import as a single element
group with no name, which is what they said. Nothing invents a
division.

One case cannot make the round trip. Exodus holds a single element type
per block, so an element group with quads and triangles in it (a
`canopy` group, for example) goes out as two blocks and comes back as
two element groups. The two blocks cannot share the id or the name the
group declared: the first keeps both, and the next takes the lowest
free id and its element type in its name (`canopy TRI3`). Blocks
sharing an id make a file that ParaView's IOSS reader cannot open (it
reports zero cells without saying why), and Visual Dynamics refuses
duplicate block ids on import.

Exodus has no traceline and needs none: a drawn line is an element
group of two-node beam elements with no properties, and it goes out as
that block and comes back as one. Coordinate systems survive as
frames, with their assignments in node sets (above), but their names
do not, since a frame is only an id, three points and a tag.

Visual Dynamics keeps no separate
tracelines: a line drawn through nodes is an element group of two-node
line elements with no properties, and that is how it is recognized on
the way out. UNV, sdynpy, ESCDF and Nastran keep lines separate from
elements, so such a group is written as the format's line (dataset 82,
the traceline array, `line_connection`, PLOTEL chains) and not also as
elements. A group that carries a section is structure, and is written
as elements. On the way in, a line arrives as its own element group,
named for the line and colored by it; a UNV line that lifts the pen
arrives as one group of several runs. An element group holds one
element family, so a source that mixes families in a group is split on
arrival, with the new groups named for their family beside the old
name. (UNV and sdynpy carry no groups at all, and a Nastran PSHELL
covers quads and triangles alike.)

UNV writes geometry as datasets 2411, 82, 2412 and 2420, and data
as one dataset 58 per record. The two coherence types keep their own
function codes, 6 for ordinary and 26 for multiple, because writing
multiple coherence as 6 would promise a reference DOF per record that
is not there. When the object's units are defined, a dataset 164 is
written declaring the unit system the values are in, and a file that
has one reads back with its units. When they are not defined, no 164
is written.

A dataset 58 names its dimension from a fixed vocabulary, and a PSD's
`acceleration**2/frequency` is not in it. Rather than write half a
dimension, the record says nothing and the data comes back unit-less.
Values, DOFs and function type all survive.

The dimension a record names is whatever the record knows: its own
when its units are defined, otherwise its hint. So a UNV that arrived
with no 164 goes back out still saying what its channels are.

## Rattlesnake runs

A Rattlesnake `.nc4` is the recording of a controller run, with its
hardware settings, environments and everything else the controller
kept, none of which Visual Dynamics holds. A file written from Visual
Dynamics would be a controller run that never happened, so the format
is import only. The controller's random specification file (`.npz`) is
different: a specification edited here can be written back out as one
(see the table). The controller's own specification files name no
channels, since it matches them to its channel table by position, so
their channels are read as '1', '2' and so on in that order, the same
as a sine specification's.

The one `.nc4` that is written is the run itself, shorter: a truncated
copy. *Save Truncated Copy…* in the long-run import window below, or
*File → Truncate Rattlesnake File…* for a run of any size, writes the
run cut to the stretch chosen. Every group, attribute, dimension and
variable is copied as the file has it, with the same storage,
compression and chunking: the hardware settings, all of the channel
fields, and each environment's control settings and specification. Only
the time data changes, each stream cut to the samples an import of the
same window reads. All channels are kept, since leaving one out would
mean rewriting the channel table and each environment's channel
indices. The file has no clock of its own, so the copy's time starts at
zero again, and the name it is offered under says where the stretch
came from (`random_30-60s.nc4` for seconds 30 to 60 of `random.nc4`).
Settings that describe the run as a whole, such as its ramp time, still
describe the original run. From a script:
`visualdynamics.io.rattlesnake.save_window(path, target, start=30,
stop=60)`, which is also the call the journal records.

Rattlesnake writes three kinds of file for a run, and all three are
read: the streamed recording, the spectral saves, and the saved
system-ID package (*Save System ID*), which holds the measured plant
FRF with its multiple coherence, plus the driven and ambient
autospectra as a pair of PSDs. The signal-to-noise is a reading of
that pair, not a third object. Select the two PSDs together and the
bar offers them *Overlaid*, as *Signal to noise (dB)* line by line, or
as *RMS signal to noise* per channel. A line where the ambient is zero
is left as a gap (NaN) rather than an infinite ratio: on real hardware
zero means below resolution, and in a simulation it means the ambient
was exactly silent. The package records hardware channel numbers but
no node names or units, so its DOFs arrive as those numbers and its
drives as 9000-series labels (9001, 9002, …), which are plainly not
real channels.

A run that controlled a virtual response (a response transformation
matrix over the control channels, as a MIMO test on a rigid fixture
often uses) imports as both halves of what the controller did. The
streamed time data stays in hardware channels, and after them, in the
same time history, come the transformation's rows over those channels,
computed as the controller computed them. Because they are one object,
one pass of averaging gives the PSDs of the raw channels and of the
virtual ones together, and the virtual ones are what the specification
was judged against. The specification, the FRF, the coherence and the
control CPSD are over those rows too, because that is what the
controller saved. The file does not name the rows, so they import as
node-only coordinates numbered from 1; rename them in the tree if the
rows mean something (a virtual point's X, Y and Z, say). A row's unit
is the control channels' unit when they all share one, and undeclared
when they do not.

A long run need not be imported whole. A stream that would take a
quarter of the machine's memory or more opens a dialog before it is
read. It shows one channel's envelope over the whole run (read in slabs
and thinned to 4,096 points, so the preview of a 22 GB run is a few
hundred kilobytes), the truncate reading's span over it to choose the
stretch, the channels to keep, and what the choice costs in samples and
gigabytes against the machine's memory. Import reads exactly that
window as a slice of the file, inclusive at both ends on the run's own
clock, which the records keep, and every row's comment names the window
and the file. A script says the same thing with
`import_file(path, start=60, stop=120, channels=['101Z+', '104Z+'])`,
and the journal replays the import as it was made. A stream the window
misses entirely is left out, a windowed stream is not split into a
spectral save's frames, and an environment's virtual responses come
along only when every control channel they are computed from is among
the channels kept. Below the threshold nothing is asked, and the API
imports the whole run unless given a window. Once the stretch is less
than the whole run, the dialog also offers *Save Truncated Copy…*,
which writes the run cut to that stretch as a `.nc4` of its own
(above).

A system ID saved as time data is a different case: the file cannot be
told apart from a run of its environment, since two streams are two
streams. When an import has exactly that shape (two streams, a quiet
one then a loud one), the app asks whether it is a system ID run
instead of guessing. If you answer yes, the project type becomes System
ID and the two recordings take their phase names, Noise Time History
and Excitation Time History, told apart by level. If you answer no,
everything keeps its generic name. Scripted imports never ask; set
`project.project_type` yourself.
