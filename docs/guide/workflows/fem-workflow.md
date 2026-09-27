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
model is checked against come from that shared data.

The whole model is also a demonstration module,
`visualdynamics.demo.barc`, so every step here can be followed exactly.

| Step | Where | How |
|---|---|---|
| Build the geometry from planes | a script | `mesh.plane(...)` per plane, `mesh.assemble(...)` |
| Tie the bolted parts | a script, or the app's add mode | a block of two-node lines made *rigid (massless)* |
| Give each part its material and thickness | the Blocks table | pick a material, type a thickness |
| Solve | the bar | *Solve Modes* |
| Compare with the reference | select both shape sets | the MAC appears; *Match Modes* commits pairs |

## 1. The geometry, from planes

A plate model is described most simply as its planes: each a rectangle
at the plate's mid-thickness, given as a corner, two edges and an element
size. `mesh.plane` meshes one evenly, with elements as close to square
and to that size as the rectangle allows, in a block named for the part;
`mesh.assemble` joins them and merges the nodes they share, so planes
meeting along a line — the box's corners, a channel's web and flanges —
are tied there. Planes given the same name are one block: the box's five
walls are the box.

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

The two channels are turned a quarter turn from each other, as the
hardware is; the lengths are the solid model's, in inches, and the
geometry holds them in SI.

## 2. The bolted joints

Where the parts are bolted, their mid-surfaces do not meet — a channel's
foot sits 0.1875 in above the box wall's mid-surface — so nothing is
shared and something has to join them. That is what a **rigid,
massless link** is: a two-node line whose second node moves exactly as
the first does, carried by its rotation, adding no mass. A block of
two-node lines given the material *rigid (massless)* is a set of them.

How the links are laid out is the modeling decision that matters most
here, and it was measured three ways against the reference:

| Joint | First ten modes against the reference |
|---|---|
| each bolt a single link at its center | 6 to 26% soft, and softer at every mesh refinement — a point tie on a plate is a local singularity |
| each foot tied over its whole area | 5 to 20% stiff |
| each bolt tied over its **washer's area** | converged, and within a few percent (below) |

So each bolt ties every node of the upper plate within its washer's
radius — 0.1875 in at the feet, 0.25 in at the beam — to the nearest node
of the plate below. `visualdynamics.demo.barc.geometry()` builds the
planes above, those ties, and gives every block its properties; save it
and open it in the app:

```python
from visualdynamics.demo import barc

barc.project().save('barc.vdyn')    # the model, and the reference to check it against
```

The same project with every step below already taken — the modes
solved and the first ten matched — is on the website's
[downloads page](https://visualdynamics.org/downloads#examples), and is
`barc.project(solved=True)`.

For a model of your own, a few links are quicker in the app: edit the
geometry's **Elements**, switch on add mode (**+**), choose the beam and,
in the **Block** drop-down beside it, *New block*; click the two nodes of
each link, and every link picked joins that block.

## 3. In the app: materials, thicknesses, solve

Open `barc.vdyn`. Expand **BARC** in the tree and click the pencil on
**Blocks**: one row per part, and to the right of Name and Elements the
property columns, in the display units.

| Block | Material | Thickness |
|---|---|---|
| box | 6061-T6 | 0.25 in |
| right channel, left channel, beam | 6061-T6 | 0.125 in |
| bolts | rigid (massless) | — |

Picking a material from the drop-down fills its modulus, Poisson's ratio
and density; any number can be typed over. A geometry meshed apart or
imported without shared corners can be tied with **Merge Coincident
Nodes** on its bar.

Select **BARC** and press **Solve Modes**; ask for 2000 Hz. The model —
5,470 nodes, 5,136 plates, 74 rigid links, 0.820 kg (the solid model's
volumes give 0.816 kg; the plates carry no bolt holes) — solves in a few
seconds, and its modes land in BARC's group: six rigid-body modes at
0 Hz and the elastic ones after them.

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
| 185.7 | 193.0 | +4.0% | 0.996 |
| 206.2 | 210.6 | +2.1% | 0.994 |
| 264.8 | 257.0 | −2.9% | 0.990 |
| 439.2 | 455.7 | +3.8% | 0.990 |
| 466.7 | 486.7 | +4.3% | 0.978 |
| 555.0 | 569.6 | +2.6% | 0.972 |
| 570.2 | 590.0 | +3.5% | 0.917 |
| 660.9 | 674.3 | +2.0% | 0.916 |
| 1083.7 | 1081.7 | −0.2% | 0.938 |
| 1147.3 | 1137.2 | −0.9% | 0.825 |

The first ten elastic modes pair one-to-one and in order, the shapes
agreeing at MAC 0.83 to 0.996 and the frequencies within −3 to +4%.
Above 1.5 kHz, where modes crowd, the pairing loosens (MAC under 0.5 for
some): the two models there are no longer describing the same modes one
for one.

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
