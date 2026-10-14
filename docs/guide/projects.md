# Projects, links, and the Basis

## A project is a mapping with structure

`visualdynamics.Project` maps a name to an object, and also holds what
the tree shows around the objects: the object groups, which group is
the Basis, the project type and the active geometry.

```python
project = visualdynamics.Project('Beam Airplane Modal Test')
project.add('Geometry', geometry)     # returns the name used
project['FRF']                        # by name, like the tree
list(project)                         # every name
print(project)                        # the tree, printed
```

A `Project` is a `dict`, so anything that takes a mapping of objects
takes a project.

## Reaching objects by what they are

A name is arbitrary. It came from an importer or from whoever named
the object, and importing a second geometry does not rename the first,
but it does make "the geometry" ambiguous. The type of an object is
not arbitrary, so the type is the better thing to write, and it is the
only one an editor can complete:

```python
project.geometry              # anywhere in the project
project.basis.frf             # in the Basis group only
project.other.shapes          # the one group that is not the Basis
project.object_group_selections[1].geometry    # any object group by position
```

The singular gives you the one object of that kind. When there are
several, it raises an error that names them, instead of returning
something whose type depends on what the project holds:

```
AttributeError: 2 in project — 'Geometry', 'FEM Geometry'.
Use .geometries[i] to pick one, or the name directly.
```

The plural is always a list, even for zero or one object, so code
written against a project with one FRF keeps working when a second
arrives:

```python
for frf in project.frfs:
    frf.plot()
```

These attributes are computed each time they are read, so adds,
renames and deletions show up immediately. `dir(project)`, and so tab
completion, lists only the kinds this project holds.

The kinds are `geometry`, `photos`, `channel_table`, `time_history`,
`transient_specification`, `spectrum`, `specification`, `psd`, `srs`,
`shock_specification`, `sine_sweep_specification`, `sine_levels`,
`frf`, `coherence`, `shapes`, `matches` and `report`, each with its
plural (`geometries`, `photo_sets`, `channel_tables`,
`time_histories`, `transient_specifications`, `spectra`,
`specifications`, `psds`, `srs_sets`, `shock_specifications`,
`sine_sweep_specifications`, `sine_level_sets`, `frfs`, `coherences`,
`shape_sets`, `match_sets`, `reports`). A broad kind does not return
an object that has a narrower kind of its own: a specification is a
PSD underneath, but `project.psds` holds only the measured PSDs, and
`project.time_history` does not return a transient specification.
[The objects](objects.md) says what each kind holds and what can be
done with it.

Names are still used by links, report bindings and the file format,
so `project['Experimental Modes']` and `project.basis.names` are
there when a name is what you mean.

## Inside a geometry

In the tree, a geometry lists its nodes, its coordinate systems, its
elements by family (beams, triangles, quads, tetras, wedges, hexes and
points, with the element groups of each family under it) and its
solids. A script reaches the nodes, coordinate systems, elements and
element groups through four views:

```python
geometry = project.basis.geometry
geometry.nodes                # <202 nodes>
geometry.coordinate_systems
geometry.elements
geometry.groups               # <3 element groups>: the parts the mesh is divided into
```

Each of these is a **view**, not a copy. Its columns are the
geometry's own arrays under plural names (`.columns` lists them), and
a row is reached by index, so both of these move a node:

```python
geometry.nodes.xyz[:, 2] += 0.5      # every node up half a meter
geometry.nodes[0].xyz = [0, 0.1, 0]  # one of them, in place
```

Iterate over a view or slice it to get rows. Adding and deleting go
through the geometry, which keeps the ids unique and the connectivity
consistent:

```python
node = geometry.nodes.add([1.0, 0.0, 0.0])   # returns the new id
geometry.elements.add([101, 106, 111])       # a triangle, in an element group of triangles
geometry.add_beams([101, 106, 111, 116])     # a drawn line: an element group of beams
geometry.nodes.delete([305])
geometry.elements.delete([12])
```

All four views delete by id. Deleting an id that does not exist is
not an error.

An element group is a part: it has an id and a name, and its contents
are read from the elements (`elements_in('wing')`,
`group_of(elem_id)`). Deleting an element group deletes the part: its
elements, and the nodes that no element of another element group
uses. Nodes it shares with another element group stay, so the
neighboring part keeps its nodes. Element groups that hold the same
element types and have the same material and thickness or section can
be merged into one (`merge_groups([7, 9])`, where the first keeps its
id and name; in the app, select them in the tree or in the Element
Groups table and press **Merge Element Groups**). Moving an element to
another element group is an edit to the *element*: in the app, the
Element group column of the element table. An element group holds one
element family, so moving a quad into an element group of beams is
refused, with the reason. An element group of two-node beams with no
properties is a **drawn line**, what other formats call a traceline;
formats that keep lines separate from elements read and write it as
one, and it becomes structure as soon as the element group is given a
section.

Node and coordinate-system ids are unique, because connectivity and
node placement refer to them. An element id is only a label.

CAD geometry is imported the same way. STEP and IGES files are
tessellated on import by a geometry kernel that the packaged builds
include and a pip install adds with the `step` extra. STEP, IGES, 3MF
and STL all come in as triangle face elements with one element group
per part, and an assembly's part names become the element group
names; 3MF and STL can also be exported. A CAD geometry is context
(the shape around the sensor points, not a set of measurement DOFs),
so its node ids are the ones the importer assigned.

## Objects compute; the project also keeps the result

A method on an object computes or draws something and returns the
result. A verb on the project does the same, and also stores the
result, names it and links it to what it came from:

```python
psds = project.basis.time_history.compute_psds()   # just the PSDs
name = project.compute_psds(project.basis.time_history)  # ...and in the project
name = project.compute_cpsds(project.basis.time_history)  # the whole matrix
```

Verbs take an object as readily as a name, so the two can be combined.

## Staleness: the project knows what a change affected

Each computation verb records how its result was computed: the verb,
the source, the parameters, and a fingerprint of the settings it read
(the averaging, the shock windows, or a digest of the source's own
content). The verbs that do this are the `compute_*` family (spectra,
PSDs, CPSDs, FRFs, multiple coherence, SRS, octave bands), filtering,
truncation, integration, differentiation, sine-level extraction, the
transforms to modal responses and back to physical ones, rigid-body
mode shapes, and authored specifications. Change the settings
afterwards and the derived objects report themselves stale, each with
the reason. A modal fit and a projection onto the Basis are judgment
calls rather than recomputations, and record nothing.

```python
history.averaging = replace(history.averaging, frames=8)
project.stale()
# {'Time History PSDs': 'computed with 4 frames of 8192 (hann); '
#                       'the recording now says 8 frames of 8192 (hann)'}
project.refresh('Time History PSDs')   # recompute in place: same name,
                                       # links and report bindings hold
project.refresh_stale()                # bring the whole chain up to date
```

Recomputing is always something you ask for, never automatic, so a
report does not change while someone is reading it. In the app the
same information is the refresh badge on the object's row, with the
reason in its tooltip. Clicking the badge recomputes the object, and
so does *Recompute* on the bar while the object is selected. The
fingerprints are settings, not timestamps, so the badges survive a
save and a load, and they appear only when the numbers would actually
differ.

## Merging

`project.merge(*names)` combines compatible objects into one. It is
meant for a survey acquired in passes, where each pass brought its own
time data, FRFs and channel table:

```python
project.merge('Pass 1 FRF', 'Pass 2 FRF', name='FRF')
```

Only objects of the same type merge, and each kind has its own rule
for what may be combined. Geometries need disjoint node and
coordinate-system ids (element ids are renumbered). Shape sets need
the same DOFs. Data arrays need an identical abscissa, and
frequency-domain arrays such as FRFs are refused when two records
share a (response, reference) pair, since the same measurement twice
is not one set. Geometries and shape sets must all have defined units
or all be raw. Channel tables merge on *identity*: a row is the same
channel when everything except its channel number matches, and the
merged table is renumbered 1 to N. Two rows for the same point and
role with different descriptions are refused, and the error names the
point, because keeping one description would drop the other without
saying so. The merged object replaces its parts and keeps their links.

Time data merges in two cases only. Repeat runs share every
channel, and each record is another average of the same setup. One
acquisition split across files shares no channel and has equal
capture counts, so the records were taken at the same time. A mixture
of the two is refused. A roving survey's passes share their references
and nothing else, and merging their time data would lose the record of
which force went with which response. Merge the passes' *FRFs*
instead, since each was computed from its own matched captures; that
is the example above.

## Links: explicit, never inferred from node numbers

Objects that describe one structure are **linked**. Shapes belong with
their geometry, and a PSD belongs with the time history it came from.
Links come from you (`link`, `set_basis`, or *Link* in the app), from
the file (a Rattlesnake run's objects arrive linked, and a `.vdyn`
keeps its object groups), and from computations, since every derived
object is linked to its source. In the app, a project with a type also
puts the objects that fill its Basis slots into the Basis group as
they arrive; the other side is never guessed. Nothing is linked by
comparing node numbers, because a finite element mesh contains almost
any test's node numbers, so node coverage does not show that two
objects belong together.

```python
project.link('Geometry', 'FRF', 'Time History')
project.link('Geometry', 'FRF', name='Pass 1')   # a named object group
project.object_group_of('FRF')       # the members linked with it
project.unlink('Time History')
```

A group holds at most one geometry, and its members are read
against that geometry. A member that names nodes the geometry does not
have is still linked, because tests name drive points, control
coordinates and virtual channels that the model does not draw. Its row
in the tree turns red, and hovering over it shows which nodes are
missing ([Compatibility](../compatibility.md)).

Computing PSDs, fitting modes and projecting a set onto another's DOFs
all link the result to what it came from.

## The Basis

One group can be declared the **Basis of comparisons**:

```python
project.set_basis('Geometry', 'FRF')     # link them and mark the group
project.set_basis('FRF')                 # or mark an existing group
project.basis                            # the members
```

The Basis group's DOFs are the space comparisons are made in (other
sets are projected onto them), its modes are the MAC rows, and its
frequencies are the baseline for Δf%. Report templates bind to it
symbolically, so a report works in any project whatever its objects
are called.

The name is generic on purpose: comparing two finite element models,
or two test runs, works the same way as comparing a test with a model.

## Which geometry an object answers to

```python
project.geometry_for('FEM Modes')      # ('FEM Geometry', <Geometry>)
```

The answer is the geometry in the object's group, or the active
geometry when the group has none. Animations, DOF arrows,
[compatibility checks](../compatibility.md) and projections all use
it, which is why a finite element shape set linked to its own mesh is
not marked inconsistent just because the test geometry is active.

## The file

```python
project.save('modal.vdyn')
back = visualdynamics.Project.open('modal.vdyn')
same = visualdynamics.import_file('modal.vdyn')     # also a Project
```

One `.vdyn` file holds the objects, the links and roles, the type and
the active geometry. `project.import_file(path)` imports *into* an
existing project. A saved project brings its structure with it, and
its object groups follow their objects when a name clashes and an
object is renamed.

## Copy and paste

Select objects in the tree and press **Cmd+C** (Mac) or **Ctrl+C**
(Windows, Linux). The selection goes on the clipboard in two forms at
once:

- Paste back into the tree (Cmd+V / Ctrl+V) and each object is
  duplicated beside the original, named `… copy`, with a number added
  if that name is taken. A copy is a separate object with its own
  arrays, and it is recorded in the console as `project.duplicate(...)`,
  the verb a script calls.
- Paste into a folder in Finder or Explorer and each object is
  written as its own file, named for the object: an export with no
  dialog. A channel table is written as a spreadsheet and a photo set
  as a folder of images; everything else is a `.vdyn`. Select the
  project row and copy to get the whole project as one file. The files
  are written only when the file manager asks for them, so copying is
  instant however large the project is.

Files copied *in* a file manager can be pasted into the tree too, and
they import the same way a drop does.
