# Projects, links, and the Basis

## A project is a mapping with structure

`visualdynamics.Project` maps a name to an object, and carries what the tree
shows around it: link groups, which group is the Basis, the project
type, the active geometry.

```python
project = visualdynamics.Project('Beam Airplane Modal Test')
project.add('Geometry', geometry)     # returns the name used
project['FRF']                        # by name, like the tree
list(project)                         # every name
print(project)                        # the tree, printed
```

Because a Project *is* a dict, anything that takes a mapping of
objects takes a project.

## Reaching objects by what they are

A name is arbitrary — it came from whatever an importer or a user
called it, and importing a second geometry does not rename the first
but does make "the geometry" ambiguous. A **type** is not arbitrary,
so that is the path worth typing, and the only one an editor can
complete:

```python
project.geometry              # anywhere in the project
project.basis.frf             # scoped to the Basis group
project.other.shapes          # the one group that is not the Basis
project.groups[1].geometry    # or any group by position
```

The **singular** gives you the one object of that kind. When there are
several it refuses and names them, rather than handing back something
whose type depends on the project's contents:

```
AttributeError: 2 in project — 'Geometry', 'FEM Geometry'.
Use .geometries[i] to pick one, or the name directly.
```

The **plural** is always a list, even for zero or one — so code
written against a project with one FRF keeps working when a second
arrives:

```python
for frf in project.frfs:
    frf.plot()
```

Nothing here is a snapshot: attributes are computed on access, so
adds, renames and deletions show up immediately. `dir(project)` — and
so tab completion — lists only the kinds this project actually holds.

The kinds are `geometry`, `photos`, `channel_table`, `time_history`,
`transient_specification`, `spectrum`, `specification`, `psd`, `srs`,
`shock_specification`, `sine_sweep_specification`, `sine_levels`,
`frf`, `coherence`, `shapes`, `matches` and `report`, each with its
plural (`geometries`, `photo_sets`, `channel_tables`,
`time_histories`, `transient_specifications`, `spectra`,
`specifications`, `psds`, `srs_sets`, `shock_specifications`,
`sine_sweep_specifications`, `sine_level_sets`, `frfs`, `coherences`,
`shape_sets`, `match_sets`, `reports`). [The objects](objects.md)
says what each one holds and what can be done with it.

Names still matter — links, report bindings and the file format all
use them — so `project['Experimental Modes']` and
`project.basis.names` are always there when a name is what you mean.

## Inside a geometry

The tree shows a geometry as five groups, and a script reaches them by
the same five names:

```python
geometry = project.basis.geometry
geometry.nodes                # <202 nodes>
geometry.coordinate_systems
geometry.tracelines
geometry.elements
geometry.blocks               # <3 blocks>: the parts the mesh is divided into
```

A group is a **view**, not a copy. Its columns are the geometry's own
arrays under plural names, and a row under its singular — so both of
these move a node:

```python
geometry.nodes.xyz[:, 2] += 0.5      # every node up half a meter
geometry.nodes[0].xyz = [0, 0.1, 0]  # one of them, in place
```

Ask a view what it holds with `.columns`; iterate or slice it for rows.
Adding and deleting go through the geometry, which is what keeps ids
unique and connectivity honest:

```python
node = geometry.nodes.add([1.0, 0.0, 0.0])   # returns the new id
geometry.tracelines.add([101, 106, 111])
geometry.nodes.delete([305])
geometry.tracelines.delete([12])
```

Everything is deleted by **id**, in all five groups. Deleting an id
that is not there is not an error.

A block is a *label on elements* rather than something with a shape of
its own: it has an id and a name, and what it holds is read off the
elements (`elements_in('wing')`, `block_of(elem_id)`). So deleting one
moves its elements into the first block left rather than taking them
with it, the last block cannot go while an element names one, and moving
an element between blocks is an edit to the *element* — in the app, the
Block column of the element table.

Node and coordinate-system ids are unique, because connectivity and
node placement refer to them. A traceline or element id is a label, and
one traceline id can name several polylines — a UNV trace line that
lifts the pen arrives split into its drawn runs, still one trace line.
Deleting that id removes the whole thing, gaps and all.

CAD geometry arrives the same way. STEP and IGES (tessellated on
import by a geometry kernel the packaged builds include and a pip
install adds with the `step` extra), 3MF and STL come in as triangle face
elements with **one block per part** — an assembly's names become the
block names, and 3MF and STL write back out. Such a geometry is
context: the shape behind the sensor points, not a set of measurement
DOFs, so its node ids are the importer's own.

## Objects compute; the project also keeps

A method **on an object** computes or draws and returns the result. A
verb **on the project** does the same and also stores it, names it,
and links it to what it came from:

```python
psds = project.basis.time_history.compute_psds()   # just the PSDs
name = project.compute_psds(project.basis.time_history)  # ...and in the project
name = project.compute_cpsds(project.basis.time_history)  # the whole matrix
```

Verbs take an object as readily as a name, so the two compose.

## Staleness: the project knows what a change touched

Every computation verb — the `compute_*` family, filtering,
truncation, integration and differentiation — records how its result
was computed: the verb, the source, the parameters, and a fingerprint
of the settings it read (the averaging, the shock windows, or a digest
of the source's own content). Change the settings afterwards and the
derived objects report themselves stale, each with the reason. (A
modal fit, a projection onto the Basis and a sine extraction are
judgment calls rather than recomputations, and record none.)

```python
history.averaging = replace(history.averaging, frames=8)
project.stale()
# {'Time History PSDs': 'computed with 4 frames of 8192 (hann); '
#                       'the recording now says 8 frames of 8192 (hann)'}
project.refresh('PSDs')        # recompute in place: same name,
                               # links and report bindings hold
project.refresh_stale()        # settle the whole chain
```

Recomputing is an act, never automatic — a report must not rewrite
itself while its author is reading it. In the app the same knowledge
is the refresh badge on the object's row, with the reason in its
tooltip; clicking the badge *is* the recompute, and so is *Recompute*
on the bar while the object is selected. The
fingerprints are settings, never timestamps, so the badges survive a
save and a load and appear exactly when the numbers would differ.

## Merging

`project.merge(*names)` combines compatible objects into one — the
tool for a survey acquired in passes, where each pass brought its own
time data, FRFs and channel table:

```python
project.merge('Pass 1 FRF', 'Pass 2 FRF', name='FRF')
```

Same concrete type only, and each kind has its own rule about what
may join: geometries need disjoint node ids, shape sets the same DOF
cover, data arrays an identical abscissa. Channel tables merge on
*identity* — a row is the same channel when everything but its
channel number matches — and the merged table renumbers 1..N; two
rows claiming the same point and role with different descriptions
refuse, naming the point, because silently keeping one description
would silently drop the other. The merged object replaces its parts
and keeps their links.

Time data merges only when the result tells a true story: **repeat
runs** (every channel shared — each record is another average of the
same setup) or **one acquisition split across files** (no channel
shared, equal capture counts — the records played at the same time).
A mixture refuses: a roving survey's passes share their references
and nothing else, and merging them would leave which-force-caused-
which-response recorded nowhere. Merge the passes' *FRFs* instead —
each was computed from its own matched captures — which is the
example above.

## Links: explicit, never inferred

Objects that describe one structure are **linked**. Shapes belong with
their geometry, a PSD belongs with the time history it came from.
Nothing in visualdynamics guesses this — a finite element mesh contains almost
any test's node numbers, so node coverage never really decided
anything.

```python
project.link('Geometry', 'FRF', 'Time History')
project.group_of('FRF')       # the members linked with it
project.unlink('Time History')
```

A group holds **at most one geometry** — its members are read against
that geometry — and a member naming nodes the geometry lacks is
refused, because the link would be a claim that is not true.

Derived objects link themselves: computing PSDs, fitting modes,
projecting a set onto another's DOFs all link the result to what it
came from.

## The Basis

One group can be declared the **Basis of comparisons**:

```python
project.set_basis('Geometry', 'FRF')     # link them and mark it
project.set_basis('FRF')                 # or mark an existing group
project.basis                            # the members
```

The Basis group's DOFs are the space comparisons happen in — other
sets project onto them — its modes are the MAC rows, and its
frequencies are the Δf% baseline. Report templates bind to it
symbolically, so a report works in any project whatever its objects
are called.

The name is deliberately generic: comparing two finite element models
or two test runs works the same way as test-against-model.

## Which geometry an object answers to

```python
project.geometry_for('FEM Modes')      # ('FEM Geometry', <Geometry>)
```

Its link group's geometry, else the active one. Animations, DOF
arrows, [compatibility checks](../compatibility.md) and projections all
ask this, which is
why a FEM shape set beside its own mesh never reads as inconsistent
just because the test geometry is active.

## The file

```python
project.save('modal.vdyn')
back = visualdynamics.Project.open('modal.vdyn')
same = visualdynamics.import_file('modal.vdyn')     # also a Project
```

One `.vdyn` file holds the objects, the links and roles, the type and
the active geometry. `project.import_file(path)` imports *into* an
existing project — a saved project brings its structure with it, and
its groups follow their objects through any name clash.

## Copy and paste

Select objects in the tree and press **Cmd+C** (Mac) or **Ctrl+C**
(Windows, Linux). The selection is on the clipboard two ways at once:

- **Paste back into the tree** (Cmd+V / Ctrl+V) and each object is
  duplicated beside its original, named `… copy`, numbered if that is
  taken. A copy is its own object — its arrays are independent — and
  it is journaled as `project.duplicate(...)`, the verb a script calls.
- **Paste into a folder** in Finder or Explorer and each object lands
  as its own file, named for the object — an export with no dialog. A
  channel table lands as a spreadsheet and a photo set as a folder of
  images, since those are what they are; everything else is a
  `.vdyn`. Select the project row and copy to get the whole project as
  one file. The files are written only when a file manager asks for
  them, so copying is instant however large the project.

Files copied *in* a file manager paste into the tree too: they import
the way a drop does.

