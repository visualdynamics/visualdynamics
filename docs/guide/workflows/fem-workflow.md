# The finite element workflow: the BARC

A finite element model built from the planes a structure is made of,
solved, and checked against a published model of the same structure —
the geometry in a short script, everything after it in the app.

The structure is the **BARC** (Box Assembly with Removable Component),
a small bolted aluminum article the structural dynamics community
shares as a common test piece: a square box tube whose top wall is
slotted in two, and on it the *Bench* — two channels standing on the two
halves and a flat bar bolted across their tops. Its solid model, test
data and a finite element model's modes are shared on the SEM Dynamic
Substructuring Focus Group wiki, <https://wiki.sem.org/wiki/BARC>. Every
dimension below comes from that solid model; the reference modes the
model is checked against come from that shared data, and are another
group's work: R. Schultz, T. Schoenherr and B. Owens, "A Proposed
Standard Random Vibration Environment for BARC and the Boundary
Condition Challenge," IMAC 2021 — the field-configuration modes of their
finite element model of the BARC, in the "Random Vibration Data"
shared by Sandia National Laboratories on the SEM Dynamic Substructuring Focus Group wiki.
The demo's project carries the same credit as its **About the Reference**
report.

Every step is in the app; the whole model is also a demonstration
module, `visualdynamics.demo.barc`, so each can be checked against it or
skipped.

| Step | Where | How |
|---|---|---|
| Start a geometry | the project row's bar | **+** *New Geometry* |
| Build it from planes | the geometry's bar | *Add Plane*, once per plane |
| Tie the bolted parts | the geometry's Elements | pick the elements under each washer, *Tie* to the part below |
| Give each part its material and thickness | the Blocks table | pick a material, type a thickness |
| Solve | the bar | *Solve Modes* |
| Compare with the reference | select both shape sets | the MAC appears; *Match Modes* commits pairs |

The reference the model is checked against comes in the demo's project,
so start there — save it and open it in the app:

```python
from visualdynamics.demo import barc

barc.project().save('barc.vdyn')    # the reference, and the demo's own model of it
```

It holds **Reference Geometry** and **Reference Modes** — the shared
data's 59 points and 30 modes — and **BARC**, the model this page builds,
already built. Build yours beside it, or skip to step 3 with BARC.

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
press **Add Plane**. Type each row below — block, corner, the two edges —
with an element size of 0.125 in, and press **Add**; the dialog stays
open for the next row, draws the plane in the 3-D view as it is typed,
and says before each Add how many of its nodes land on nodes already
there (the right wall's 25 on the bottom's edge, for one).

| Plane | Block | Corner | Edge A | Edge B |
|---|---|---|---|---|
| box, bottom | box | (−2.875, −2.875, 0) | (5.75, 0, 0) | (0, 0, 3) |
| box, right side | box | (2.875, −2.875, 0) | (0, 5.75, 0) | (0, 0, 3) |
| box, left side | box | (−2.875, −2.875, 0) | (0, 5.75, 0) | (0, 0, 3) |
| box, top left of the slot | box | (−2.875, 2.875, 0) | (2.625, 0, 0) | (0, 0, 3) |
| box, top right of it | box | (0.25, 2.875, 0) | (2.625, 0, 0) | (0, 0, 3) |
| right channel, foot | right channel | (1.5, 3.0625, 1) | (0.9375, 0, 0) | (0, 0, 1) |
| right channel, top | right channel | (1.5, 4.9375, 1) | (0.9375, 0, 0) | (0, 0, 1) |
| right channel, web | right channel | (2.4375, 3.0625, 1) | (0, 1.875, 0) | (0, 0, 1) |
| left channel, foot | left channel | (−2.5, 3.0625, 1.0625) | (1, 0, 0) | (0, 0, 0.9375) |
| left channel, top | left channel | (−2.5, 4.9375, 1.0625) | (1, 0, 0) | (0, 0, 0.9375) |
| left channel, web | left channel | (−2.5, 3.0625, 1.0625) | (1, 0, 0) | (0, 1.875, 0) |
| beam | beam | (−2.5, 5.0625, 1) | (5, 0, 0) | (0, 0, 1) |

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
here, and it was measured four ways against the reference:

| Joint | First ten modes against the reference |
|---|---|
| each bolt a single link at its center | 6 to 26% soft, and softer at every mesh refinement — a point tie on a plate is a local singularity |
| each foot tied over its whole area | 5 to 20% stiff |
| each bolt tied at every node within its washer's radius | −3 to +4%, MAC 0.83 to 0.996 |
| each bolt tied over **the elements its washer covers** | 0 to +6%, MAC 0.945 to 0.995 — what is built here |

The last two are both the washer's area, and both converge: halving the
mesh moves the frequencies about 1% for the first and 0.4 to 2.5% for
the second. The elements are what
is built because they are what a person picks: whole elements, countable
in the view.

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

In the app: click the pencil on the geometry's **Elements**. For each
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
solved and the first ten matched — is on the website's
[downloads page](https://visualdynamics.org/downloads#examples), and is
`barc.project(solved=True)`.

## 3. Materials, thicknesses, solve

Expand the geometry in the tree — yours, or **BARC** — and click the pencil on
**Blocks**: one row per part, and to the right of Name and Elements the
property columns, in the display units.

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

## 4. Compare with the reference

Select **Reference Modes** and the solved modes together. The comparison
screen reads the MAC between them, projecting the model's modes onto the
reference's 59 points — its sensors are on the parts' surfaces and the
model's nodes on their mid-surfaces, half a thickness away, which the
projection allows for. Pick the matching squares and press **+** to
commit the pairs; the matched-modes table gives each pair's frequency
error and MAC.

| Reference (Hz) | Model (Hz) | Difference | MAC |
|---|---|---|---|
| 185.7 | 194.5 | +4.8% | 0.995 |
| 206.2 | 212.5 | +3.0% | 0.995 |
| 264.8 | 265.3 | +0.2% | 0.989 |
| 439.2 | 461.8 | +5.1% | 0.964 |
| 466.7 | 494.9 | +6.1% | 0.945 |
| 555.0 | 581.0 | +4.7% | 0.985 |
| 570.2 | 593.5 | +4.1% | 0.969 |
| 660.9 | 681.4 | +3.1% | 0.946 |
| 1083.7 | 1128.1 | +4.1% | 0.990 |
| 1147.3 | 1170.9 | +2.1% | 0.991 |

The first ten elastic modes pair one-to-one and in order, the shapes
agreeing at MAC 0.945 to 0.995 and the frequencies 0 to 6% above the
reference's; the next two pair as well (1503.6 and 1609.2 Hz, MAC 0.89
and 0.99). Above 1.7 kHz, where modes crowd, the pairing loosens (MAC
under 0.5 for some): the two models there are no longer describing the
same modes one for one.

## What the check does and does not say

The model is a few percent stiff, consistently, and it is left that way
rather than tuned: the reference is a different kind of model (not
plates), its material values are not published, and a check tuned to
agree says nothing. The shared data's own notes also warn that the
physical hardware differs from the drawing — the box's wall thickness
most of all — so against a measurement of a particular unit, measure the
unit and model what was measured.

From a script, the same comparison is one call:

```python
from visualdynamics.demo import barc

model = barc.build()
shapes = model.eigensolution(maximum_frequency=2000)
for row in barc.compare(shapes, model):
    print(row['reference'], row['model'], row['error'], row['mac'])
```
