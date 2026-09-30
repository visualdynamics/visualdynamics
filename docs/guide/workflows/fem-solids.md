# The solid workflow: the four-unit frame

A finite element model built from the blocks a thick part is made of,
meshed into solid bricks with its holes cut and its fillets kept, tied
where it is screwed, and solved. All of it in the app.

The structure is the **four-unit frame**, a small aluminum ladder the
structural dynamics community shares as a test article for
substructuring: a 16 by 6 in frame cut from half-inch plate, two rails
and five uprights half an inch wide, 41 threaded steel inserts to bolt
things to, and two rectangular 22 by 4.4 in wings, thin (1/8 in) and
thick (1/4 in), that screw across two of its uprights. Its solid
models, finite element models and test data are shared on the SEM
Dynamic Substructuring Focus Group wiki,
<https://wiki.sem.org/wiki/Round_Robin_Frame_Structure>. Every dimension
below was read off the finite element models shared there, and the
models were checked against the frequencies measured there.

A half-inch bar spanning a hundred-millimeter window is neither a beam
nor a plate. The [BARC](fem-workflow.md) is built from planes of plate
elements because its walls are thin; this frame is built from **blocks
of eight-node solid bricks**, the element written for it: trilinear
bricks with incompatible bending modes, so one brick through a plate's
thickness bends like the plate, and a stubby member is what it is.

Every step is in the app; the whole model is also a demonstration
module, `visualdynamics.demo.frame`, so each can be checked against it
or skipped:

```python
from visualdynamics.demo import frame

frame.project().save('frame.vdyn')    # the five models this page builds
```

| Step | Where | How |
|---|---|---|
| Start a geometry | the project row's bar | **+** *New Geometry* |
| Build it from blocks | the geometry's bar | *Add Block*, once per block |
| Cut the holes | a script | `mesh.block(..., holes=...)`, which the dialog does not type |
| Tie the screwed parts | the geometry's Elements | pick the wing's bricks under each washer, *Tie* to the frame |
| Give each block its material | the Blocks table | a material; nothing else for a block of solids |
| Solve | the geometry's bar | *Solve Modes*, to 1500 Hz |

## 1. Blocks

Set the display units to inches (the unit menu on the toolbar). Start a
geometry and add blocks: each is a corner, three perpendicular edges and
an element size, in the display unit, drawn in the 3-D view as it is
typed. A block whose face falls on a face already there shares its
nodes, so blocks that meet are tied where they meet. Give blocks
that meet the same size, and make the face they share the same
rectangle on both sides: a rail meshed whole puts its nodes between an
upright's, so the demonstration meshes each rail as the segments
between the uprights. The frame's top face is at z = 0 with the plate
below it, as the shared models have it.

| Block | Corner | Edge A | Edge B | Edge C |
|---|---|---|---|---|
| frame, a rail | (−8, 2.5, −0.5) | (16, 0, 0) | (0, 0.5, 0) | (0, 0, 0.5) |
| frame, the other rail | (−8, −3, −0.5) | (16, 0, 0) | (0, 0.5, 0) | (0, 0, 0.5) |
| frame, an upright at x | (x − 0.25, −2.5, −0.5) | (0.5, 0, 0) | (0, 5, 0) | (0, 0, 0.5) |

The uprights stand at x = 0, ±3.875 and ±7.75. A twelfth of an inch
puts six bricks across a member and three across an insert hole; a
coarser size solves faster and lands a few percent further from the
measurements.

From a script the same blocks are `mesh.block` and `mesh.assemble` (or
`project.add_block`, which is what each Add records), and a script can
say what the dialog does not: the **holes**. Each is a center, a radius
and the axis it runs along, blind to a depth or through, and either a
void or the name of the block its bricks go to, such as a threaded
insert given its own material. The bricks on a hole's rim are moved onto its
circle, so a hole is round and not stair-stepped. The frame's 41
inserts are each a 0.29 in hole drilled through, with the insert filling
its top 5/16 in:

```python
import numpy as np
from visualdynamics import mesh

x, y, z = np.eye(3)
size = 1 / 12                                     # inches
holes = []
for cx, cy in frame.INSERTS:                      # 41 of them
    holes.append(((cx, cy, 0), 0.1451, 2, None, None))         # drilled through
    holes.append(((cx, cy, 0), 0.1451, 2, 0.3125, 'inserts'))  # the insert in its top
rail = mesh.block((-8, 2.5, -0.5), 16 * x, 0.5 * y, 0.5 * z, size, 'frame',
                  unit='in', holes=holes)
```

The fillets where an upright meets a rail are small square blocks with
a void of the fillet's radius cut from their far corner; what is left
is the material in the corner of the window. They matter: without them
the frame's in-plane modes come out eight percent low.

## 2. The wings and their screws

A wing is one block, 4.385 by 22 in, its thickness the wing's, set
0.028 in above the frame (the washer under it) across the uprights at
x = 0 and 3.875, with its eighteen 0.209 in holes cut through. Four
screws hold it: at each, pick the wing's bricks within the washer's
radius of the screw, the full thickness of the wing, and **Tie** them
to the frame. The rigid links are the screw and its washers clamping
the plate; the demonstration lumps each screw's few grams at the frame
under it.

## 3. Materials and solve

Expand the geometry in the tree and click the pencil on **Blocks**: one
row per part. A block of solids takes a material and nothing else.

| Block | Material |
|---|---|
| frame | 6061-T6 |
| inserts | the insert material the shared model states, spread over the hole |
| thin wing, thick wing | 6061-T6 |
| screws | rigid (massless), set by Tie |

Select a geometry and press **Solve Modes**. The frame (33,600 nodes,
23,800 bricks, 632 g against the 622 to 626 g of the frames that were
weighed) solves in half a minute, and its modes land in the geometry's
group: six rigid-body modes at 0 Hz and the elastic ones after them.

## What was checked

Two things, and they are different checks.

**The element**, against the finite element model shared on the wiki:
that mesh, read with the Nastran importer (its PSOLID and MAT1 cards
become the blocks' materials) and solved here, gives the first ten
elastic modes MSC Nastran gives it to within a tenth of a percent, on
265,000 degrees of freedom, in two minutes and eight gigabytes. That is the
brick element and the sparse solver agreeing with a solver the audience
trusts, on the same mesh.

**The model built here**, against what was measured. The wiki lists
the first nine modes of five frames, which differ among themselves by
up to five percent; the frame built above lands within one percent of
frame SN003 on six of them and within four on the three in-plane modes
its stair-stepped fillets soften. The wings' frequencies were fitted
from the shared test FRFs with *Fit Modes*: the thin wing lands within
one percent of every measured mode, the thick wing within three. The
model is left as it is rather than tuned toward any of them; a check
tuned to agree says nothing.

| Frame mode | Measured, SN003 | Shared model | Built here |
|---|---|---|---|
| 1 | 234.2 | 234.2 | 234.7 |
| 2 | 291.7 | 296.3 | 294.6 |
| 3 | 624.4 | 627.6 | 628.4 |
| 4 | 652.8 | 659.2 | 657.4 |
| 5 | 715.3 | 715.5 | 697.8 |
| 6 | 752.0 | 747.7 | 725.4 |
| 7 | 1118.8 | 1127.8 | 1124.1 |
| 8 | 1155.3 | 1165.6 | 1165.4 |
| 9 | 1241.3 | 1244.6 | 1215.7 |

None of the wiki's files are in this package or its examples; the
numbers above are what was read from them.

From a script:

```python
from visualdynamics.demo import frame

model = frame.build()                         # the frame alone
shapes = model.eigensolution(maximum_frequency=1500)
print(shapes.frequency[6:15])                 # the first nine elastic modes
both = frame.build('thick wing')              # the frame with a wing on it
```
