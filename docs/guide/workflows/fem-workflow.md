# The finite element workflow: the BARC

A finite element model built from the planes a structure is made of,
tied where it is bolted, and solved — all of it in the app. (A part too
thick for plates is built from blocks of solid bricks instead: [the
solid workflow](fem-solids.md), on the four-unit frame.)

The structure is the **BARC** (Box Assembly with Removable Component),
a small bolted aluminum article the structural dynamics community
shares as a common test piece: a square box tube whose top wall is
slotted in two, and on it the *Bench* — two channels standing on the two
halves and a flat bar bolted across their tops. Its solid model, test
data and finite element models are shared on the SEM Dynamic
Substructuring Focus Group wiki, <https://wiki.sem.org/wiki/BARC>. Every
dimension below comes from that solid model, and the model was checked
against the finite element models shared there.

Every step is in the app; the whole model is also a demonstration
module, `visualdynamics.demo.barc`, so each can be checked against it or
skipped:

```python
from visualdynamics.demo import barc

barc.project().save('barc.vdyn')    # the model this page builds, already built
```

| Step | Where | How |
|---|---|---|
| Start a geometry | the project row's bar | **+** *New Geometry* |
| Build it from planes | the geometry's bar | *Add Plane*, once per plane |
| Tie the bolted parts | the geometry's Quads | pick the elements under each washer, *Tie* to the part below |
| Give each part its material and thickness | the pencil on each block's row | pick a material, type a thickness |
| Solve | the bar | *Solve Modes* |

## 1. The geometry, from planes

A plate model is described most simply as its planes: each a rectangle
at the plate's mid-thickness, given as a corner, two edges and an element
size. Each edge is divided evenly into the whole number of elements
nearest that size, so the elements come out close to square. A plane's
nodes that fall on nodes already in the geometry become them, so planes
meeting along a line — the box's corners, a channel's web and flanges —
are tied there; and planes given the same block name are one block: the
box's five walls are the box.

Choose the inch system in the unit menu, select the project's row and
press **+** (*New Geometry*): an empty geometry, in inches, named
*Geometry*. Select it and
press **Add Plane**. A pane opens beside the 3-D view. Type each row
below — block, center, widths — with an element size of 0.125 in, and
press **Add**. A plate leaves one width at zero, the axis it faces: the
bottom's zero width along Y puts it in the X-Z plane. The pane stays
open for the next row, draws the plate in the 3-D view as it is typed,
and says before each Add how many of its nodes land on nodes already
there (the right wall's 25 on the bottom's edge, for one). The rings
around the preview turn it about its center and the arrows slide the
center along its axes, onto a tenth of an inch; the table's rows need
neither.

| Plane | Block | Center | Widths |
|---|---|---|---|
| box, bottom | box | (0, −2.875, 1.5) | (5.75, 0, 3) |
| box, right side | box | (2.875, 0, 1.5) | (0, 5.75, 3) |
| box, left side | box | (−2.875, 0, 1.5) | (0, 5.75, 3) |
| box, top left of the slot | box | (−1.5625, 2.875, 1.5) | (2.625, 0, 3) |
| box, top right of it | box | (1.5625, 2.875, 1.5) | (2.625, 0, 3) |
| right channel, foot | right channel | (1.96875, 3.0625, 1.5) | (0.9375, 0, 1) |
| right channel, top | right channel | (1.96875, 4.9375, 1.5) | (0.9375, 0, 1) |
| right channel, web | right channel | (2.4375, 4, 1.5) | (0, 1.875, 1) |
| left channel, foot | left channel | (−2, 3.0625, 1.53125) | (1, 0, 0.9375) |
| left channel, top | left channel | (−2, 4.9375, 1.53125) | (1, 0, 0.9375) |
| left channel, web | left channel | (−2, 4, 1.0625) | (1, 1.875, 0) |
| beam | beam | (0, 5.0625, 1.5) | (5, 0, 1) |

The two channels are turned a quarter turn from each other, as the
hardware is; the lengths are the solid model's, in inches (the geometry
holds them in SI). Up is +y, as in the solid model.

From a script the same planes are `mesh.plane` and `mesh.assemble` (or
`project.add_plane`, which is what each Add records):

```python
import numpy as np
from visualdynamics import mesh

x, y, z = np.eye(3)
size = 0.125                                 # inches

def plate(corner, a, b, name):
    return mesh.plane(corner, a, b, size, name, unit='in')

t = 2.875                                    # the box wall's mid-surface
barc = mesh.assemble(
    plate((-t, -t, 0), 2 * t * x, 3 * z, 'box'),            # bottom
    plate((t, -t, 0), 2 * t * y, 3 * z, 'box'),             # right side
    plate((-t, -t, 0), 2 * t * y, 3 * z, 'box'),            # left side
    plate((-t, t, 0), (t - 0.25) * x, 3 * z, 'box'),        # top, left of the slot
    plate((0.25, t, 0), (t - 0.25) * x, 3 * z, 'box'),      # top, right of it
    plate((1.5, 3.0625, 1), 0.9375 * x, z, 'right channel'),       # its foot
    plate((1.5, 4.9375, 1), 0.9375 * x, z, 'right channel'),       # its top
    plate((2.4375, 3.0625, 1), 1.875 * y, z, 'right channel'),     # its web
    plate((-2.5, 3.0625, 1.0625), x, 0.9375 * z, 'left channel'),
    plate((-2.5, 4.9375, 1.0625), x, 0.9375 * z, 'left channel'),
    plate((-2.5, 3.0625, 1.0625), x, 1.875 * y, 'left channel'),
    plate((-2.5, 5.0625, 1), 5 * x, z, 'beam'),
)
```

## 2. The bolted joints

Where the parts are bolted, their mid-surfaces do not meet — a channel's
foot sits 0.1875 in above the box wall's mid-surface — so nothing is
shared and something has to join them. That is what a **rigid,
massless link** is: a two-node line whose second node moves exactly as
the first does, carried by its rotation, adding no mass. A block of
two-node lines given the material *rigid (massless)* is a set of them.

How the links are laid out is the modeling decision that matters most
here, and it was settled by checking the model against the wiki's: a
single link at each bolt comes out soft, and softer at every mesh
refinement — a point tie on a plate is a local singularity, not a
joint — and each foot tied over its whole area comes out stiff. Tied
over the washer's area, the model agrees and converges. The elements
the washer covers are what is built, rather than every node within its
radius, because elements are what a person picks: whole elements,
countable in the view.

So under each bolt the elements its washer covers are tied, every node
of them, to the nearest nodes of the part below. There are ten bolts,
four under each channel's foot and one at each end of the beam; at the
0.125 in mesh the patches are:

| Bolts | Where, (x, z) | The elements under each | Tie to |
|---|---|---|---|
| left foot | (−2.1875, 1.3125), (−2.1875, 1.6875), (−1.8125, 1.3125), (−1.8125, 1.6875) | the element the bolt passes through and the eight around it | box |
| right foot | (1.8125, 1.3125), (1.8125, 1.6875), (2.1875, 1.3125), (2.1875, 1.6875) | the element the bolt passes through and the eight around it | box |
| beam | (−2, 1.5), (2, 1.5) | the 4 × 4 around the node the bolt passes through | the channel under it |

On a foot, whose mesh is 8 × 8, the four patches are the 6 × 6 inside
its outer ring of elements, split into four 3 × 3 squares. On the beam,
8 elements wide, each patch is the middle four across its width, the
third to sixth elements from its end.

In the app: expand the geometry and click the pencil on **Quads**. For each
bolt, click one element of its patch in the 3-D view and Shift-click
(Cmd-click on a Mac) the rest; then press **Tie** on the bar and pick
the part below from its menu — *box* for a foot, *left channel* or
*right channel* for the beam. Every node of the patch is linked to the
nearest node of that part, and the links go into a block of rigid,
massless links that the first Tie makes (*ties*) and every later one
joins. Ten Ties make the 178 links. (For a joint where the nearest node
of a whole part might be the wrong one, *A second selection…* in the
same menu ties the patch to a second patch picked next.)

The same project with every step below already taken — the modes
solved — is on the website's
[downloads page](https://visualdynamics.org/downloads#examples), and is
`barc.project(solved=True)`.

## 3. Materials, thicknesses, solve

Expand the geometry in the tree — yours, or **BARC** — then **Quads**,
and click the pencil on a block's row: the blocks table opens on it,
one row per part, and to the right of Name and Elements the property
columns, in the display units.

| Block | Material | Thickness |
|---|---|---|
| box | 6061-T6 | 0.25 in |
| right channel, left channel, beam | 6061-T6 | 0.125 in |
| ties (*bolts* in the demo's BARC) | rigid (massless), set by Tie | — |

Picking a material from the drop-down fills its modulus, Poisson's ratio
and density; any number can be typed over. A geometry meshed apart or
imported without shared corners can be tied with **Merge Coincident
Nodes** on its bar.

Select the geometry and press **Solve Modes**; ask for 2000 Hz. The model —
5,470 nodes, 5,136 plates, 178 rigid links, 0.820 kg (the solid model's
volumes give 0.816 kg; the plates carry no bolt holes) — solves in a few
seconds, and its modes land in the geometry's group: six rigid-body
modes at 0 Hz and the elastic ones after them. Planes typed from the
table above are the demo's node for node, and the patches above are the
demo's element for element, so a geometry built by hand solves to the
same modes.

## What was checked

While the model was built, its modes were checked against the finite
element models shared on the [wiki](https://wiki.sem.org/wiki/BARC);
the joints above are the ones that agreed with them. The model is left
as it is rather than tuned toward any one of them — a check tuned to
agree says nothing — and against a measurement of a particular unit,
measure the unit and model what was measured: hardware differs from its
drawing, the box's wall thickness most of all.

From a script:

```python
from visualdynamics.demo import barc

model = barc.build()
shapes = model.eigensolution(maximum_frequency=2000)
print(shapes.frequency[6:16])                 # the first ten elastic modes
```

## The BARC from bricks

The same structure is also built from solid bricks, the way the
[four-unit frame](fem-solids.md) is: each part as the boxes it is made
of, meshed into eight-node bricks. It is the higher-fidelity model, and
it comes with the removable component (the two channels and the beam)
as a model of its own. The joints follow the finite element model shared
on the wiki. The channels' feet share nodes with the box over their
whole footprint. The beam rests 0.001 in over the channels, so it touches
them only where it is tied, over each washer. Each bolt is a point mass
at its head: a block of point elements given a mass in the Blocks table.

```python
from visualdynamics.demo import barc

barc.solid_project(solved=True).save('barc-bricks.vdyn')  # both, solved
```

At an eighth of an inch the assembly has 15,267 nodes and solves in
about ten seconds. Its first ten elastic modes land 1 to 4 % above the
ones measured on the BARC and shared on the wiki, and its mode shapes
follow the shared model's in order. Leaving the bolt masses out raises
the modes by up to 8 %.
