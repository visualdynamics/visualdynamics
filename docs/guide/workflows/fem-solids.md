# The solid workflow: the Frame-Wing

A finite element model of solid bricks, built in the app the way a part
is drawn: solid shapes added, copied in rows, subtracted and joined,
then meshed, the parts bolted together by merging nodes, and solved.
Everything on this page is done with the mouse and the keyboard; the
demonstration module builds the same models with the same steps.

The structure is the **Frame-Wing**, a small aluminum frame and two
wings the structural dynamics community shares as a test article for
substructuring: a 16 by 6 in ladder cut from half-inch plate, two rails
and five uprights half an inch wide, 41 threaded steel inserts to bolt
things to, and two rectangular 22 by 4.385 in wings, thin (1/8 in) and
thick (1/4 in), that screw across two of the frame's uprights. Its
solid models, finite element models and test data are shared on the SEM
Dynamic Substructuring Focus Group wiki,
<https://wiki.sem.org/wiki/Round_Robin_Frame_Structure>. Every dimension
below was read off the finite element models shared there.

A half-inch bar spanning a hundred-millimeter window is neither a beam
nor a plate. The [BARC](fem-workflow.md) is built from planes of plate
elements because its walls are thin; this frame is built from
eight-node solid bricks with incompatible bending modes, so one brick
through a plate's thickness bends like the plate.

```python
from visualdynamics.demo import frame_wing

frame_wing.project().save('frame-wing.vdyn')    # the five models this page builds
```

The project's journal (the console) is this page's steps, one line per
click.

| Step | Where | How |
|---|---|---|
| Start a geometry | the project row's bar | **+** *New Geometry* |
| Add the shapes | the geometry's **Solids** row | *Add Solid Block*, *Add Cylinder* |
| Pattern them in rows | solids picked in the tree | *Pattern* |
| Cut and join them | solids picked in the tree | *Subtract*, *Union* |
| Mesh | parts picked in the tree | *Mesh Solids* |
| Bolt a wing on | element groups picked in the tree | *Align Meshes*, then *Merge Coincident Nodes* |
| Give each part its material | the pencil on an element group's row | the Element Groups table |
| Solve | the geometry's bar | *Solve Modes* |

## 1. The frame, as solids

Choose the inch system in the unit menu, select the project's row and
press **+** (*New Geometry*); rename it *Frame*. Select its **Solids**
row and add each shape below with *Add Solid Block* or *Add Cylinder*:
the pane beside the 3-D view takes a center, the widths (a block) or a
radius and a length along Z (a cylinder), a corner radius (a block) and
a name, draws the shape as it is typed, and **Add** puts it in the tree.

| Shape | Add | Center | Size | Name |
|---|---|---|---|---|
| plate | Solid Block | (0, 0, −0.25) | 16 × 6 × 0.5 | frame |
| window | Solid Block | (−5.8125, 0, −0.25) | 3.375 × 5 × 1, corner 0.138 | |
| rail hole | Cylinder | (−7.75, −2.75, −0.25) | radius 0.1451, length 1 | |
| upright hole | Cylinder | (−7.75, −1.375, −0.25) | radius 0.1451, length 1 | |

The window is cut deeper than the plate so it cuts all the way through, and
its corner radius rounds its four edges along Z: cut from the plate, it
leaves the 0.138 in fillets the frame has in each window's corners.
Without them the frame's in-plane modes come out eight percent low.

Then pattern them. Pick a solid in the tree and press *Pattern*: how
many along X, Y and Z, the original included, and how far apart, the
copies drawn as they are typed. **Pattern** makes one solid, the
pattern, with the original under it in the tree, and picks it for the
next step. The copies follow the original: its pencil changes every
copy, and the pattern's own pencil changes how many there are and how
far apart. One pattern is a whole grid, so both rails' 26 holes are
one hole and 13 along X by 2 along Y.

| Pick | How many along X, Y, Z | Spacing along X, Y, Z |
|---|---|---|
| the window | 4, 1, 1 | 3.875, 0, 0 |
| the rail hole | 13, 2, 1 | 1.291667, 5.5, 0 |
| the upright hole | 5, 3, 1 | 3.875, 1.375, 0 |

That is four windows, 26 holes along the rails and 15 down the
uprights, 41 in all, in three patterns. Now pick the plate and the
three patterns (click the plate, Shift-click the last pattern) and
press *Subtract*: it takes the others from the first in the tree, the
plate, and its tooltip says so. The result is named for the plate,
*frame*.

The inserts fill the top 5/16 in of each hole. Add the same two
cylinders again, centered a little higher and shorter, named *inserts*,
and pattern them the same way:

| Shape | Add | Center | Size | Name |
|---|---|---|---|---|
| rail insert | Cylinder | (−7.75, −2.75, −0.15625) | radius 0.1451, length 0.3125 | inserts |
| upright insert | Cylinder | (−7.75, −1.375, −0.15625) | radius 0.1451, length 0.3125 | inserts |

Pick the two patterns and press *Union*: one part, *inserts*.

## 2. Mesh

Pick *frame* and *inserts* in the tree and press *Mesh Solids*. Type
0.083333 for the element size (a twelfth of an inch: six bricks across
a member, three across a hole) and press **Mesh**. Each part becomes an
element group under **Hexes**, the two sharing their nodes around each
hole. Each hole is meshed as rings of bricks about its circle, with a
square core where an insert fills it, and the fillets are rounded:
33,320 bricks, 45,087 nodes.

## 3. Materials and solve

Click the pencil on an element group's row: the Element Groups table
opens, one row per part. An element group of solids takes a material
and nothing else.

| Element group | Material |
|---|---|
| frame | 6061-T6 |
| inserts | typed: modulus 1.1216e6 psi, density 3.6175e-4 lbf·s²/in⁴, Poisson's ratio 0.3 |

The inserts' numbers are the shared model's: a stiffness tuned to the
tested frames and a density that gives the frame its measured mass,
spread from the insert's annulus over its whole hole. Select the
geometry and press **Solve Modes**; it solves in about half a minute,
six rigid-body modes at 0 Hz and the elastic ones after them.

## 4. The wings

Each wing alone is a geometry of its own: a plate with eighteen holes,
1 in long cylinders through it. For the thick wing (the thin one is
the same with 0.1243 for the thickness and 0.08975 for every Z):

| Shape | Add | Center | Size | Name |
|---|---|---|---|---|
| wing | Solid Block | (1.9375, 0, 0.15575) | 4.385 × 22 × 0.2563 | thick wing |
| screw hole | Cylinder | (0, −2.75, 0.15575) | radius 0.1043, length 1 | |
| rail hole | Cylinder | (1.2917, −2.75, 0.15575) | radius 0.1043, length 1 | |
| end hole | Cylinder | (0.5626, −7.75, 0.15575) | radius 0.1043, length 1 | |

| Pick | How many along X, Y, Z | Spacing along X, Y, Z |
|---|---|---|
| the screw hole | 2, 5, 1 | 3.875, 1.375, 0 |
| the rail hole | 2, 2, 1 | 1.291667, 5.5, 0 |
| the end hole | 2, 2, 1 | 2.75, 15.5, 0 |

Subtract the three patterns, eighteen holes, from the wing, mesh it
at 0.125, and give it 6061-T6. It sits 0.0276 in above where the frame's top would be: the
washers under it on the frame.

## 5. A wing bolted to the frame

The wing on the frame is one geometry: the frame's steps again (copy
the *Frame* geometry before meshing it, with Copy, then Paste on the
project, and rename the copy *Thick Wing on Frame*), then the screws,
then the wing's steps, then the bolting.

### The screws

Each screw is threaded into the insert under it, and
the steel there is what makes the joint stiff. In the copy, add a
cylinder over one insert at a screw, named *screws*, and pattern it
into the four, a pattern named *screws* as the cylinder is:

| Shape | Add | Center | Size | Name |
|---|---|---|---|---|
| screw | Cylinder | (0, −2.75, −0.15625) | radius 0.1451, length 0.3125 | screws |

| Pick | How many along X, Y, Z | Spacing along X, Y, Z |
|---|---|---|
| the screw | 2, 2, 1 | 3.875, 5.5, 0 |

Then mesh *frame*, *inserts* and *screws* together at 0.083333. Where
two parts overlap, the later in the tree takes the bricks, so the four
inserts under the screws are steel.

### The wing and its washers

Add the wing's shapes (section 4) and a
washer under it at each screw, patterned the same way:

| Shape | Add | Center | Size | Name |
|---|---|---|---|---|
| washer | Cylinder | (0, −2.75, 0.0138) | radius 0.196, length 0.0276 | washers |

| Pick | How many along X, Y, Z | Spacing along X, Y, Z |
|---|---|---|
| the washer | 2, 2, 1 | 3.875, 5.5, 0 |

Pick the wing and the washers, press *Mesh Solids*, type 0.125, tick
**Keep apart from other groups**, and press **Mesh**: the washers share
nodes with the wing above them, and their undersides lie on the
frame's top without sharing a node with it.

### Bolting

The washers and the frame were meshed apart, so the
washers' nodes do not land on the frame's. Under **Hexes**, pick the
element groups *frame*, *inserts*, *screws* and *washers* and press
*Align Meshes*: each node of the frame's top under a washer is marked
with a line to the washer node it would move onto (the washers' rings
are the finer mesh, so the frame's nodes are the ones that move).
Press **Align**.
With the same four picked, press *Merge Coincident Nodes* and
**Merge**: the washers are joined to the frame at the places just
aligned, and the wing is bolted on.

### The screws' heads

Each screw's grams, its head and the washer
under the head, sit on the wing's top: click the pencil on the
**Points** row, switch on add mode (**+**), choose *New element group*,
and click the node of the wing's top nearest each screw. Name the
group *screw heads* and give it a mass of 1.9986e-5 lbf·s²/in (3.5 g)
in the Element Groups table, and give *screws* and *washers* 1018
steel.

Select the geometry and press **Solve Modes**. The thick wing on the
frame is 65,325 nodes and solves in about a minute.

## What was checked

Three different checks were made.

### The element

The finite element model of the frame shared on the wiki was read with
the Nastran importer and solved here. Its first ten elastic modes match
the ones MSC Nastran gives for the same mesh to within a tenth of a
percent, which checks the brick element and the sparse solver apart
from the meshing.

### The frame and the wings, against test

The wiki lists the first nine modes of five frames, which differ among themselves by
up to five percent; the frame built here lands within 1 % of frame
SN003 on six of them and within 4 % on the three in-plane modes. The
wings' frequencies were fitted from the shared test FRFs with *Fit
Modes*: the thin wing lands within 1 % of every one, the thick wing
within 3 %.

| Frame mode | Measured, SN003 | Built here |
|---|---|---|
| 1 | 234.2 | 235.1 |
| 2 | 291.7 | 291.7 |
| 3 | 624.4 | 628.8 |
| 4 | 652.8 | 652.9 |
| 5 | 715.3 | 696.0 |
| 6 | 752.0 | 725.1 |
| 7 | 1118.8 | 1124.5 |
| 8 | 1155.3 | 1160.9 |
| 9 | 1241.3 | 1214.6 |

### The wings on the frame, against the shared models

The finite element models of the same assemblies are shared on the
wiki, tuned there to test (Linderholt, 2024). Against them, every one
of the first nine modes built here is within 3 %. The shared models
join each wing with rigid links between the screws' parts, which their
author notes may make them slightly stiffer.

| Mode | Thick wing on frame, shared | Built here | Thin wing on frame, shared | Built here |
|---|---|---|---|---|
| 1 | 126.0 | 125.2 | 63.8 | 62.4 |
| 2 | 233.4 | 240.1 | 113.5 | 111.8 |
| 3 | 242.7 | 243.3 | 224.4 | 222.8 |
| 4 | 312.4 | 309.1 | 224.8 | 225.8 |
| 5 | 439.3 | 440.0 | 268.6 | 271.3 |
| 6 | 454.3 | 453.2 | 306.2 | 303.2 |
| 7 | 460.8 | 467.2 | 373.0 | 367.0 |
| 8 | 645.1 | 654.5 | 417.2 | 412.0 |
| 9 | 676.5 | 681.7 | 497.2 | 493.7 |

A washer merged onto the insert's spread-out material read the thick
wing's first mode 7 % low; the steel screw threaded into the insert,
as the hardware has it, is what closed that gap. No property was
adjusted to bring these numbers closer. None of the wiki's
files are in this package or its examples; the numbers above are what
was read from them.

From a script:

```python
from visualdynamics.demo import frame_wing

model = frame_wing.build()                         # the frame alone
shapes = model.eigensolution(maximum_frequency=1500)
print(shapes.frequency[6:15])                      # the first nine elastic modes
both = frame_wing.build('thick wing')              # the frame with a wing on it
```
