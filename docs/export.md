# Importing and exporting

Anything visualdynamics reads *as test data*, it can also write — a
tool that only imports is a dead end for whoever needs the data next.
The read-only rows below are the deliberate exceptions: solver results
(a Nastran punch, a Femap neutral) and controller output (Rattlesnake)
are other programs' outputs, and writing one would mean impersonating
the program that makes them.

**`.vdyn` (HDF5) is the native format**, and the only one that keeps
everything — units included. Save and Load use it. It is provisional
for the alpha (the next alpha saves `.escdf`; [its own
page](vdyn-format.md) says what that means for a file you have). It
is standard, documented HDF5 — the layout is that page, and
any HDF5 tool reads one without this package. The foreign formats
below exist to exchange data with other tools, and each loses whatever
it has no way to record.

One row per format, R/W meaning both directions:

| format | ext | Geometry | ShapeSet | function data | ChannelTable |
|--------|-----|----------|----------|---------------|--------------|
| Universal file ⁴ | `.unv` (read also as `.uff`) | R/W | R/W | R/W | — |
| I-DEAS ADF ² | `.afu`/`.ati`/`.ash` | — | R/W | R/W | — |
| Exodus ¹ | `.exo` (read also as `.e`) | R/W | R/W | R/W | — |
| Nastran bulk ³ | `.bdf`/`.dat`/`.nas` | R/W | — | — | — |
| Nastran punch ³ | `.pch` | — | R | — | — |
| Femap Neutral ³ | `.neu` | R | R | — | — |
| sdynpy | `.npz`/`.npy` | R/W | R/W | R/W | — |
| Rattlesnake | `.nc4` | — | — | R | R |
| Excel | `.xlsx` | — | — | — | R/W |
| 3MF CAD mesh ⁵ | `.3mf` | R/W | — | — | — |
| STL mesh ⁵ | `.stl` | R/W | — | — | — |
| STEP / IGES ⁶ | `.step`/`.stp`/`.iges`/`.igs` | R | — | — | — |
| MATLAB ⁷ | `.mat` | R/W | R/W | R/W | R/W |

Three more go in and out whole rather than by object: the project file
itself (`.vdyn`, everything a project holds — or `.mat`, the same
layout as MATLAB structs, see ⁷), a report template
(`.vdreport`, a report on its own, to load into another project — see
[reports](guide/reports.md)), and photographs (a folder of images,
`.png`/`.jpg`/`.heic`, through the project's Photos).

"Function data" is the dataset-58 vocabulary throughout: time
histories, FRFs, PSDs, spectra, coherence, multiple coherence and
SRS — though not every format carries every kind (exodus takes time
histories and spectra riding a mesh, ADF splits functions and time
data across `.afu` and `.ati`).

A channel table is the one visualdynamics object that *is* a spreadsheet, so it gets
a spreadsheet format: every column in order, channel numbers as integers,
the rest as text. Its units are metadata, not values — nothing converts.

¹ Mode shapes go to exodus **riding a geometry** — the file is a mesh with
results (one time step per mode, the frequency as the step's time), which
is what ParaView animates (its reader assembles DispX/Y/Z into a `Disp`
vector for Warp By Vector). The GUI supplies the **active geometry** (the
one marked in bold with a bullet by Set as Active Geometry), or the project's only one; with several
geometries, Set as Active is how to choose, the status line names which
was used, and a geometry covering none of the shape DOFs is refused.
Shapes alone are refused, because a mesh-less modal exodus is a file
nothing can draw. Damping and modal mass ride as two global variables
(`Damping`, `ModalMass` — one number per step, which is what a global
variable is); a complex shape goes out as its real part under the plain
name, so ParaView still assembles its `Disp` vector, with the imaginary
part beside it as `DispX_IM` and so on, and the reader puts the two back
together; DOFs at nodes the mesh lacks are dropped. The geometry's
coordinate systems go out as the file's *coordinate frames* (an id,
three points and a rectangular/cylindrical/spherical tag) and come back
as systems — their definitions, with no names. Nothing in exodus lets a
node or a variable refer to a frame, so shape values at a node with a
local displacement system are rotated to global on the way out; which
node is placed or measured in which system rides as named node sets
(`def_cs_7`, `disp_cs_7`), a standard feature every exodus tool shows
harmlessly and this reader restores the assignment from. The whole
modal round trip is lossless through Visual Dynamics; another tool
reads the mesh, the modes and the frames, and sees the sets as sets.

Time histories and spectra ride a geometry the same way: the abscissa
becomes the file's step axis, each (quantity, direction) a nodal
variable named so it reads back as itself (`AccX`, `Pressure`), a
complex spectrum as `_RE`/`_IM` pairs, DOF-less records as global
variables named by their block. A (node, variable) with no record is
NaN, never zero — an absent reading is not a reading of nothing — and
values in a node's local displacement frame rotate to the global
frame, exactly as shapes do. A stacked object (averages)
is refused with the collision named: one variable per node cannot
hold twenty readings of one channel, so pick records first. Reading
such a file back is `load(steps='time')` or `load(steps='frequency')`
— the file cannot say what its own step axis means, so the reader is
told; the default reading stays one-mode-per-step. Exodus records no
units either way: values go out in the display system and come back
raw, the variable's stem kept as a quantity hint.

² **ADF** — the I-DEAS Associated Data Files that Siemens
NX read natively. Written files were verified by reading them back
against the reference tooling, value for value, complex
shapes and complex
modal mass and damping included. Two properties are the format's own
and no writer can change them: **ADFs store SI regardless of the
display system** (the one exception to the display-units rule below —
the format defines it, an SI-writing dataset-164 in binary), and
**ADFs store float32 only**, so double-precision data narrows on the
way out. Shape files densify each node to whole X/Y/Z (or 6-DOF)
vectors, zeros at directions the set does not carry — and the
importer's zero-column policy drops those again on the way back in,
so the round trip is stable. The format layout the readers and
writers follow was derived from sample files and held to them by test;
the derivation record itself is kept, not published.

The same three extensions **import**, alongside everything else the
window accepts — and since the file is SI by construction, ADF data
arrives with units already declared, not hinted. The exception is a
record typed Unknown or General: it bypassed the SI conversion, its
values are raw, and its ordinate units label — free text, and often
wrong in real files — arrives as a **hint only**, naming the axis and
narrowing the Imported Units shortlist. Declaring the units stays a
human decision, because nothing is scaled by a guess.

**SEP 005** — sdypy's unified-timeseries standard — is exchanged at
the API rather than as a file, since it is an in-memory dict format:
`visualdynamics.from_sep005(...)` reads the dicts a sdypy package
hands over, `TimeHistory.to_sep005(...)` produces them, and the
exported form passes sdypy's own compliance validator.

³ **The model side** — the formats that carry a finite element model
and its modes, so a test column can stand beside a model column:

- **Nastran bulk data** reads and writes the mesh — grids (all three
  field formats, bare-exponent floats included), CORD2 chains,
  elements, concentrated masses, PLOTELs as tracelines. Constraints
  and analysis cards are skipped knowingly on the way in; the written
  deck is interchange, not a runnable model — no properties or
  materials are invented, and the header says who wrote it.
- **A Nastran punch file** brings the modes: SOL 103 eigenvectors
  read into a ShapeSet, frequencies from the eigenvalues. Complex
  output refuses by name until a real file shows its header layout.
- **Femap Neutral** reads both halves from one file — geometry
  (blocks 403/404) and normal-modes output (450 with 451 or 1051) —
  against layouts pinned on real files; a vintage whose record shape
  differs refuses naming the version rather than being misread.
  Rigid elements and nodes in local coordinate systems refuse by
  name; static and stress vectors are skipped knowingly.

Modes imported from a punch or a neutral arrive **unscaled**:
neither format records its normalization convention, so shapes and
MACs are exact and nothing downstream reads a physical scale out of
modal masses that were never in the file. Each module's docstring
(`io/nastran.py`, `io/punch.py`, `io/femap.py`) holds the full list
of judgment calls. OP2 is planned but gated on a real file. Abaqus
ODB is declined: a proprietary binary whose only practical reader is
Abaqus's own API, and anyone with Abaqus can export a `.fil` or a
report instead. The BK Connect and Vibrata formats are declined too —
proprietary, with no public specification and no sample ever seen
here. All three wait on a real file with a real need rather than
speculative reverse-engineering.

⁴ **The universal file has two encodings, and both are read.** An
ordinary dataset 58 writes its values as `13.5E` text; **dataset 58b**
keeps the same eleven ASCII header records and then writes the values
as raw IEEE 754 floats. The binary form is roughly a third smaller and,
unlike the text form, exact — five decimal places do not survive a
round trip and raw doubles do. LMS Test.Lab and I-DEAS both emit it.

Reading takes either, in either byte order, at either precision, and
tells the two apart by the `b` on the dataset's own marker line. Only
IEEE 754 floats are decoded: dataset 58b can also declare DEC VMS or
IBM 5/370 floating point, which are different *representations* rather
than different byte orders, and reading one as IEEE would return
plausible wrong numbers instead of failing — so the reader names the
format it found and stops. The spec is mirrored at
`docs/uff_spec/58b.asc`.

⁵ Already-tessellated CAD hand-off: export to either from SolidWorks or any CAD tool. 3MF keeps each part as a named block and declares its unit; STL is one unnamed, unit-less mesh (a block per `solid` in the rare ASCII form). Faces only — a geometry without face elements is refused. On mesh density: this geometry is context behind the sensor points, so coarse export settings are the right habit — a fine tessellation costs drawing time and buys nothing.

⁶ B-rep CAD, tessellated on import by the bundled OpenCASCADE kernel
(LGPL-2.1 with exception — one more replaceable library, see
NOTICE.md): named parts become named blocks, coordinates convert to
meters from whatever the file declares. Import only — Visual Dynamics
holds meshes, not surfaces, so writing B-rep back would be an
invention. A pip install keeps the kernel optional
(`pip install 'visualdynamics[step]'`); the packaged application
carries it.

⁷ **MATLAB** is the project file in MATLAB's container, not a foreign
format: a `.mat` written here holds exactly what a `.vdyn` holds, under
the same names ([the `.vdyn` layout](vdyn-format.md)), so nothing is
lost either way and every object kind — the project itself included —
goes out and comes back. `load('modal.mat')` gives `objects`, a cell of
structs each with its `name` and `kind`, alongside the project's
`test_name`, `project_type`, `links` and `provenance`; a single object
saved alone is one struct named for its kind (`data`, `geometry`,
`shapes`, …). The respellings MATLAB needs are mechanical: ragged
connectivity is a cell array of row vectors, one-dimensional datasets
are column vectors, strings are char arrays and lists of them cellstr,
`''` still means undeclared. **Values are SI with every unit named**,
as in `.vdyn` — the file is the project, not a display of it — so
convert in MATLAB from the unit strings if a display system is wanted.
The reader takes back the files it writes, and one other thing: a
single struct built by hand to the documented layout of one object,
without the schema stamp. Anything else in a `.mat` is refused by name
rather than guessed at. Version 5 files only (MATLAB's `-v7`), which
hold at most 4 GB per variable; a larger record is refused before
anything is written, and the `.vdyn` — which MATLAB's `h5read` opens
directly — is the file to hand over instead.

Writing takes the text form by default, because it is the one every
reader takes. The binary form is `save(obj, path, binary=True)`, or
the `unv_binary` exporter — `export_file(obj, path, format='unv_binary')`,
and its own entry in the GUI's export dialog. Only the functions change
form; geometry, mode shapes and the units dataset have no binary
variant.

`export_file(obj, path)` picks the exporter by suffix; `format=` names one
outright. The GUI offers only the formats the selected object can be
written as — mode shapes with no geometry beside them have no exodus
form, and saying so in the dialog beats saying it in an error
afterwards.

## Units

**Exports are written in the display unit system.** What is on screen is
what lands in the file: with in-lbf-s selected, a model one inch across is
written as `1.0`, not as `0.0254`. That holds for every format **except
ADF**, whose definition requires SI on disk whatever the writing session
displayed — every ADF-reading tool converts on its own side, so the
numbers still arrive right. For the rest, the same export never means
different magnitudes depending on which one you picked.

Only UNV can *say* which units it used — a dataset 164 declaring the system
and its conversion factors, so the file comes home to exactly the same SI
values whatever system it was written in. sdynpy and exodus record nothing,
so they simply carry the numbers you were looking at; the export status
line names the units, because it is the only place they are ever stated.

**An object with no units declared is written exactly as it stands.** There
is no factor to convert it by, and multiplying by one would amount to
claiming it was SI. No 164 is written and it re-imports unit-less, leaving
the next reader in the same honest position.

Naming a quantity is not sizing it, though, and a dataset 58 can do the
first without the second. A record that knows it holds accelerations —
because that is what the file it came from said, kept in `dimension_hint` —
goes out as data type 12 with no 164 to scale it, and reads back knowing the
same thing. Only a record that knows nothing writes data type 0.

Scripting default: `export_file` without a `unit_system` writes the stored
values, which are SI once units are declared.

## What each format drops

**sdynpy** keeps everything else: geometry including coordinate systems,
tracelines and elements; data including reference DOFs; shapes including
frequency, damping, modal mass and the DOF set. Where visualdynamics has both a
`comment` (from the file it came from) and a `description` (typed in
visualdynamics), the format has one field — what the user typed wins.

**Mode shapes** go to UNV as one dataset 55 per mode — the standard nodal
mode-shape record, carrying frequency, modal mass and damping ratio with
the vector at each node. Exodus stores them the way it stores any nodal
result: one "time step" per mode, with the mode's frequency as the step's
time value, and nodal variables `DispX`/`DispY`/`DispZ` (plus `RotX`/`RotY`
/`RotZ` when the shape carries rotations). Damping and modal mass ride
as the file's *global variables* `Damping` and `ModalMass`, one value
per step, so a modal file comes back with everything it left with.

Dataset 55 stores a whole vector per node, so a shape covering only some
directions is padded with zeros and comes back covering all of them.

### UNV keeps local frames; exodus keeps them beside the values

The two formats differ in what they can say, so they are treated
differently rather than uniformly.

**UNV keeps the local frames.** Dataset 55 means a node's own displacement
system by the values it holds, so the values are written as they stand and
a dataset 2420 is written alongside defining every coordinate system —
label, type, name and its 4x3 transform. Node references (`node_def_cs`,
`node_disp_cs`) come from dataset 2411 as before. A file exported this way
round trips with its frames intact.

*Caveat:* the published spec for dataset 2420 could not be reached when
this was written, and still cannot: the SDRL index lists it, but
`sdrluff/files/2420.asc` is a 404 (checked 2026-08-25 — the datasets
mirrored under `docs/uff_spec/` *do* still resolve there, so the
earlier note that the whole site had gone is wrong). The layout used
here round trips through visualdynamics but has not been checked
against the standard or against a file from another tool. If you have
the spec, drop it in `docs/uff_spec/` and it can be verified.

### Exodus results are written in the global frame

Exodus can hold coordinate frames — and since 2026-09-12 the geometry's
systems go out as them — but nothing in the format lets a node or a
variable *refer* to one, so a shape value left in a node's own
displacement frame would be read back as though it were global — wrong,
silently. Values are rotated into global on the way out, translations as a
triple and rotations as another, using the same convention the animator
uses. That needs the geometry, since the shapes alone do not say what frame
they are in: pass `geometry=` to `exodus.save`, which also writes the mesh
alongside. Without it the values go out as they stand. Which node was
placed or measured in which frame is not lost: it rides as the
`def_cs_<id>` / `disp_cs_<id>` node sets described above, and the
reader restores the assignment from them, so the systems, the
assignments and the (global) values together round-trip the model
exactly.

### Element blocks survive the round trip

An exodus mesh is written as named blocks, and the block is where a part
identity lives — `wing`, `tail`, `nacelle front left`. A geometry carries
them (`elem_block` per element, `block_id`/`block_name` declaring them),
so a file read and written back says what it said, which it did not
before: every element used to come back in one unnamed block.

They are not only for the round trip. `fem.Model.from_geometry` reads the
section of a member off the block of the element it came from, so a saved
geometry is a complete description of a structure — `sections={'prop':
BLADE}` and nothing else passed alongside the file. Asking a node instead
does not work: a node on a seam between two blocks belongs to both and
can answer for only one.

Formats that do not record the question import as a single block with no
name, which is what they said. Nothing invents a division.

One case does not round-trip, and cannot: exodus holds a single element
type per block, so a block with quads *and* triangles in it — the drone's
`canopy` — goes out as two, and comes back as two. They may share neither
the id nor the name the block declared. The first piece keeps both; the
next takes the lowest free id and its element type in the name (`canopy
TRI3`). Writing them as one id was a file nothing would read: ParaView's
IOSS reader fails it at `REQUEST_INFORMATION` and reports zero cells,
without saying why, and Visual Dynamics refuses the duplicate ids
outright. `vtkExodusIIReader` read it happily, so only the IOSS reader
catches this.

**exodus** has no traceline. Tracelines are written as runs of
two-node beam elements rather than dropped, which is lossy in one
direction: they read back as elements, because nothing in the file
says a beam was ever a traceline. Pass `tracelines_as_beams=False` to
leave them out. Coordinate systems survive — as frames, with their
assignments riding as node sets (above) — but their *names* do not: a
frame is an id, three points and a tag.

**UNV** writes geometry as datasets 2411, 82, 2412 and 2420, and data as
one dataset 58 per record. Both coherence types keep their own function
codes — 6 for ordinary, 26 for multiple — because writing multiple
coherence as 6 promises a reference DOF per record that is not there. It is the only foreign format that can carry units:
a dataset 164 declaring SI is written **when the object's units are
defined**, and omitted when they are not. Stored values are SI either way,
but only in the first case is that a fact rather than an assumption — and a
file that says so reads back with its units.

A dataset 58 names its dimension from a fixed vocabulary, and a PSD's
`acceleration**2/frequency` is not in it. Rather than write half a
dimension, the record says nothing and the data returns unit-less. Values,
DOFs and function type all survive.

The dimension it names is whatever the record knows — its own when its units
are defined, otherwise its hint. So a UNV that arrived with no 164 goes back
out still saying what its channels are.


## Rattlesnake is read-only, on purpose

A `.nc4` is the recording of a controller run — hardware settings,
environments, the lot — none of which visualdynamics holds. A file written from visualdynamics
would be a controller run that never happened. It stays an importer.

The row covers all three files a run produces: the streamed
recording, the spectral saves, and the saved **system-ID package**
(Save System ID) — the measured plant FRF with its multiple
coherence, plus the driven and ambient autospectra as a pair of
PSDs. The signal-to-noise is a *reading* of that pair, not a third
object: select the two PSDs together and the bar offers them
overlaid or divided, the ratio in decibels, with a zero ambient
line answering a gap rather than infinity — on real hardware zero
means below resolution, in a simulation it means the quiet was
exactly silent. The package records hardware channel numbers but no
node names or units, so its DOFs arrive as those numbers and its
drives as visibly synthetic 9000-series labels.

A run that controlled a **virtual response** — a response
transformation matrix over the control channels, as a MIMO test on a
rigid fixture often does — imports as both halves of what the
controller did. The streamed time data stays in hardware channels;
beside it comes a second time history, the transformation's rows over
those channels (`Random_transformed`), computed the way the
controller computed them and carrying the run's averaging, so the
PSDs made from it are what the specification was judged against. The
specification, the FRF, the coherence and the control CPSD are over
those rows too, because that is what the controller saved. The file
names the rows nowhere, so they import as node-only coordinates
numbered from 1 — rename them in the tree if the rows mean something
(a virtual point's X, Y and Z, say) — and a row's unit is the control
channels' unit when they all share one, undeclared when they do not.

A system ID saved as *time data* is a different case: the file is
indistinguishable from a run of its environment — two streams are
two streams. When an import has exactly that shape (two streams, a
quiet one then a loud one), the app asks whether it is a system ID
run rather than guessing; answered yes, the project types System ID
and the two recordings take their phase names — **Noise Time
History** and **Excitation Time History**, told apart by level —
and answered no, everything keeps its generic name. Scripted imports never
ask — set `project.project_type` yourself.
