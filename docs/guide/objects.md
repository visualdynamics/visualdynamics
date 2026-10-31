# The objects

A project holds objects, and each one is a plain Python object with
arrays in it. The window shows them, and a script holds the same
objects and calls the same methods. This page lists the kinds of
object, what each one stores, where it comes from and what can be done
with it. The API reference has the signatures; each heading below links
to its page.

Two rules apply to every kind:

- Values are stored in SI once their units are known. A record whose
  units were never declared keeps the file's raw numbers and says so
  (`ordinate_dim == 'unknown'`), and nothing is scaled by a guess.
  [Units](units.md) has the details.
- A project reaches an object by its kind (`project.frf`,
  `project.time_histories`), as [Projects](projects.md) explains.
  `project.verbs(obj)` lists the processing verbs that apply to an
  object, each with a one-line description, and
  `project.selection_verbs(*names)` lists the ones a selection of
  several objects can take together. Both lists and the acts on the
  window's bar are read from one table, so they always agree.

## Geometry — [`core.geometry`](../api/visualdynamics.core.geometry.md)

Kind `geometry`. Nodes, coordinate systems, elements and element
groups, held as flat arrays: `node_id`, `node_xyz` (meters), the system
each node is placed in (`node_def_cs`) and measured in
(`node_disp_cs`), `cs_matrix` (three direction rows and an origin per
system), one connectivity array per element (`elem_conn`), and the
element group each element belongs to (`elem_group`) with the group
names (`group_name`). `nodes`, `coordinate_systems`, `elements` and
`groups` are row views onto those arrays and are what a script usually
uses; writing through a row writes the array.
`geometry.place_coordinate_system(cs_id, angles=(x, y, z), origin=...)`
turns or moves a coordinate system, with the angles in degrees about
the fixed X, then Y, then Z axes. It is the same call the 3-D view's
rings and arrows make when a turn or a slide ends. The coordinate
systems table shows those angles as Rotation X, Y and Z, beside the
origin, and an angle typed there turns the system. X and Z read from
-180° to 180° and Y from -90° to 90°. At a quarter turn about Y, the X
and Z turns are about the same axis, so the table gives the whole turn
as Z and shows X as zero.

There are no tracelines. A line drawn through nodes is an element
group of two-node beam elements with no properties: `add_beams` chains
one through the nodes, `drawn_lines` reads them back as the runs they
were drawn in, and the same group becomes structure once it is given a
section. An element group holds one element family (beams, triangles,
quads, tetras, wedges, hexes, points or pyramids; `element_family`
names the family of a type code), which is what Exodus calls an element
block. A source that mixes families in one group is split by family
when it is read.

A geometry comes from universal files, Exodus files, Nastran bulk data
and Femap neutral files, STEP and IGES, 3MF and STL meshes, sdynpy
geometry files, MATLAB files in the project layout, ESCDF files, and
`.vdyn` files. Its methods: `plot`, `plot_dofs` and `plot_channels`; `add_node`,
`add_nodes`, `add_element`, `add_elements`, `add_beams`, `add_group`
and `add_coordinate_system`; `delete_nodes`, `delete_elements`,
`delete_groups` and `delete_coordinate_systems`; `renumber_node`,
`renumber_group` and `renumber_coordinate_system`; `missing_dofs`
against a data object; `extent`; `validate`; `save`. Plates, blocks
and the other solids are added and meshed through the project
(`project.add_solid_plate`, `project.mesh_solids` and the rest),
described below.

### Rigid-body modes and the default view

A geometry carries `mass_properties`: the reference point its
rigid-body modes pivot on and, when the set is to be mass-normalized,
the mass and inertia tensor about that point
([`core.rigid.MassProperties`](../api/visualdynamics.core.rigid.md)).
They are stored on the geometry the same way averaging is stored on a
time history. They are set in the Modes pane (the *Modes* toggle on
the 3-D view's bar, offered when one geometry is selected, with
*Kinematic* chosen in the pane) and saved with the geometry, and `suggest_mass_properties`
starts the reference point at the centroid with no mass. *Generate
Rigid Body Mode Shapes* (`project.generate_rigid_body_modes`) makes
the six-mode shape set in the geometry's group, and the modes are
previewed on the model while you set the reference point.

A geometry also carries its default `view`, which is how it opens in
3-D ([`View`](../api/visualdynamics.core.geometry.md)): the direction
from the model to the eye and which way is up. There is no distance or
zoom, since every view is fitted to the model. The app opens the
geometry and everything drawn on it (shapes, an ODS, the DOF arrows) in
this view, *Reset View* on the 3-D view's bar returns to it, and the
report's scenes and the exported figures (`plot`, `plot_dofs`, the
animations) are drawn from it. In the app, turn the model the way you
want it and press *Set Default View* on the geometry's bar. From a
script:

```python
from visualdynamics import View

project.set_view('BARC', View(eye=(1, 1, -1), up=(0, 1, 0)))  # built y-up
```

`None` returns to the default, from +X+Y+Z with Z up. Only how the
model is viewed changes; its nodes are not rotated. The view is saved
with the geometry.

### How a geometry is drawn

A solid element group is drawn as its outer skin, the faces no two of
its elements share, see-through at half strength, so a bore or a
pocket inside the part shows through the skin, and so does the part's
far side.
Plates draw as themselves, opaque from either side, and lines as
lines. Nodes are drawn as points only where the point is the
information: a node no element uses, the nodes of lines and point
elements, and whatever is picked. The corners of plates and solids
are where their mesh lines meet and are not drawn again, and a brick
model's nodes, most of which are inside it, are not drawn at all.
Picking the Nodes row in the tree draws every node, picking one node
draws that one, and hovering over a node captions it whether it is
drawn or not. The same rule draws the view, the report's scenes and
every exported figure.

### Element group properties

A geometry is also a finite element model once its element groups say
what they are made of. The pencil on an element group's row in the tree
opens the Element Groups table on that group. With the table open,
Cmd-click (Ctrl-click elsewhere) more groups' rows in the tree to add
their columns, or click a picked one off to take its column away;
picking several and choosing Edit from the right-click menu opens the
table on all of them. The table runs
across: each group is a column, named for it, and its properties read
down the column, a material (name, E, ν, ρ) and either a thickness,
for a group of quads or triangles, or a section and an orientation
vector, for a group of beams. A group of solid elements takes the
material alone, a group of points takes a mass, ground or a stiffness,
and a group of lines can take a stiffness in place of a section. A
property that a group's elements do not use is blank in that group's
column and takes no value: the Thickness row is live under a group of
quads and blank under a group of beams, and hovering over a blank cell
says why. A paste, a fill or a batch edit across several groups sets
the cells that apply, skips the rest and says on the status line how
many it skipped and why. A value left from before in a cell that does
not apply still shows, so it can be cleared, and clearing a section's
cells (its Shape, or its name) takes the section off once nothing of it
is left. Meshing solids into a group
that carries a section, a thickness or an orientation takes them off
and keeps its material, and the status line says what went. To set one value on several groups at once, select
the property's cells across their columns and batch edit, or drag the
fill handle across from the group that has it; every table convention
holds this way up.

A section is built from its shape: round tube, rod, rectangle,
rectangular tube, I-beam, channel or angle, picked in the Shape column.
The Dimensions cell then shows what the shape needs (`D=?, t=?` for a
round tube) and computes A, Iy, Iz and the torsion constant J from the
values typed. The *custom* shape takes those four numbers typed
directly. Every shape puts its width or flanges along the section's
local y and its depth along local z, and the orientation vector gives
local y, so for an I-beam or a channel it points across the flanges and
Iy is the strong axis. An angle's leg axes are not principal axes, so
its Iy and Iz are its principal moments, and its Dimensions cell says
where to point the orientation (45° from the long leg for an equal
angle). The shear center of a channel or an angle is off the centroid,
and the twisting that causes is not modeled. A rolled channel's tapered
flanges make its weak-axis Iz lower than the uniform-flange value given
here.

All of these values are shown and typed in the current display unit
system (psi and in⁴ in the inch system, MPa and mm⁴ in millimeters),
with the unit in the column header, and held in SI. The table follows
the unit menu while it is open. The Material cell is a drop-down over a
small library: 6061-T6, 7075-T6 and 2024-T3 aluminum; 1018, A36 and
4130 steel; 304, 316 and 17-4 PH stainless; Ti-6Al-4V; AZ31B magnesium;
C26000 brass; C11000 copper; Inconel 718; acrylic (PMMA);
polycarbonate; ABS; and nylon 6/6. Picking one fills the row with the
handbook's typical room-temperature values (`fem.material(name)` in a
script; `fem.MATERIAL_LIBRARY` carries a note for each entry). They are
typical values, not the values for your part: if you have a
certification, type its numbers over them. Any name typed into the
cell is accepted as a name.

An element group of two-node lines can instead be made rigid
(massless), the last entry of the Material list. Each line is then a
rigid link: its second node moves exactly as its first does (translated
by the first node's rotation about it), and it adds no mass. This is
how to model a bolt joining two plates whose mid-surfaces do not meet.
Such a group needs no section, and its modulus, density and Poisson's
ratio are blank. In the app a link is a beam element: edit the
geometry's elements, switch on add mode (**+**), choose the beam and,
in the *Element group* drop-down beside it, *New element group*. Click
the two nodes of each link; every link picked goes into that new group.
Then give the group *rigid (massless)* in the Element Groups table. The
drop-down lists the element groups of the family being added to, plus a
new one, and opens on the group the editing started from, so a beam is
never added to a group of plates.

A bolted joint is quicker as a patch. Select the elements under the
washer (click one, Shift-click the rest) and press *Tie* on the bar.
Every node of the patch is linked to the nearest node of the element
group picked from its menu, or of a second patch picked next, and the
links go into the geometry's rigid element group, which is created the
first time (`project.tie_elements`, `mesh.tie`). Plates connect only
where they share nodes, so a structure built from planes is tied along
the lines where its planes meet.

An element group of point elements takes a mass alone, typed in the
Mass column, and each element in it puts that mass at its node. Use it
for a bolt, a sensor or a fitting too small to mesh
(`fem.GroupProperties(mass=...)` in a script). A Nastran deck's plain
CONM2 cards are read this way, one element group per distinct mass,
and a deck written back carries the masses on CONM2 cards.

A group of two-node lines can be springs instead of beams. Type a
stiffness by direction in its Stiffness cell (`Kz=1e5, Kry=2e3`, in the
display units, along and about the global axes) and each line becomes
a spring between its two nodes in each direction given. The two nodes
may coincide, as they do for a joint between two parts meshed to the
same point. A group of points given a stiffness is a set of springs
from each node to ground. A group of points given *ground (fixed)* from
the Material list holds each node in all six directions, which is a
support or the far end of a spring to ground; the Ground column can
name the directions held instead (`X, Y, Z` pins a node and leaves it
free to rotate; `all` is all six). A Nastran deck's CELAS springs and
SPC supports are read as such groups, and a deck written back carries
them as CELAS2 and SPC1 cards.

Nodes in one place are drawn as one. Hovering there captions every id
(`104, 200 (2 here)`), and Shift-clicking the spot again while adding
an element picks the node not picked yet, so a spring between two
coincident nodes takes two clicks. A point mass on spring lines needs
no beam at its node. The **Points** family is always in the tree, so the
first point can be clicked into place. In a script, use
`fem.GroupProperties(stiffness=(...))` and
`fem.GroupProperties(ground=True)`, or `Model.add_spring` and
`Model.add_ground` on a model directly.

The properties are stored in `geometry.group_properties` as
[`fem.GroupProperties`](../api/visualdynamics.core.fem.md) and are
saved in the `.vdyn` file. *Solve Modes* (`project.solve_modes`) then
builds the model from the element groups (`fem.Model.from_geometry`)
and adds its normal modes to the geometry's group. It is in the Modes
pane: turn on *Modes* on the 3-D view's bar, choose *Finite element*
(offered once the element groups have properties), and set how far to
solve and the damping to give every mode. *Solve to* is either a
*Number of modes*, the lowest that many with the rigid-body modes
included, or a *Highest frequency*, every mode up to it; the status bar
says how many modes were solved and the highest frequency reached. A
script passes `num_modes=` or `maximum_frequency=`. Either way the
solve is one pass: for a frequency, the modes below it are counted
first. The pane counts the nodes, elements and
DOFs to be solved, and names any element group still without
properties. The model is free-free unless
ground groups hold it, and a free-free solution has its six rigid-body
modes at 0 Hz first. An element group with no properties, or
properties of the wrong kind for its elements, is refused by name.

### Building a mesh

*New Geometry* (the **+** on the project row's bar) starts an empty
geometry (`project.new_geometry`). Its parts are built on its
**Solids** row and meshed with *Mesh Solids*, both described under
[Solids](#solids) below: *Add Plate* and *Add Disk* for parts meshed
into shells, *Add Block* and the other shapes for parts meshed
into bricks or tetrahedra. Each part becomes an element group named for
it, and the new nodes that fall on nodes already there become those
nodes. The BARC is built this way, twelve plates meshed into quads
([the finite element workflow](workflows/fem-workflow.md)).

In a script, [`mesh`](../api/visualdynamics.core.mesh.md) also builds a
mesh directly: `mesh.plane` meshes a rectangle of plates from a corner,
two edges and an element size, `mesh.block` a box of eight-node bricks,
with cylindrical holes along one of its edges, and `mesh.join` and
`mesh.assemble` put the pieces together, sharing the nodes where they
meet.

### Merging and aligning meshes

*Merge Coincident Nodes* joins parts that were meshed apart. Pick the
element groups to join in the tree, one or several, and it is on their
bar. It opens a pane beside the view and marks every place where their
nodes coincide; all places are marked to begin with. Click a mark to
leave that place out and click it again to put it back; *All* and
*None* mark every place or none. The tolerance is typed in the display
unit and the marks follow it; it opens at a quarter of the groups'
finest mesh spacing. The picked groups are drawn see-through while the
pane is open, so a mark inside a solid is visible. *Merge* makes each
marked place one node. A rigid link, spring or beam whose nodes all
become one node no longer connects anything and is removed with them,
and the pane says so before you press *Merge*. The pane stays open on
the same groups afterward: the places merged are drawn as blue squares
and are no longer marks, the places left out are marked again for the
next *Merge*, and the pane reports what the last merge did, until you
press *Close*. Only the nodes the picked groups' elements use are
candidates, so a part left out stays separate even where it touches;
the geometry's own row does not offer the act. From a script,
`project.merge_coincident_nodes(name, tolerance, groups=[...],
exclude=[...])`, where `exclude` names the nodes of the places left out
(`Geometry.coincident_spots` lists the places). Without `groups` it
merges the whole geometry.

*Align Meshes*, beside it on the same bar, is the step before a merge
when parts meshed apart do not match. A wing meshed at an eighth of an
inch on a frame meshed at a twelfth has no coincident nodes to merge.
*Align Meshes* moves each node of the coarser mesh that lies within its
reach of the finer mesh's outside onto the finer mesh's nearest node
there. No node moves more than about half an element plus any gap
between the parts, and each moved node is then a place to merge. Its
pane is the same as the merge's, with a reach in place of the tolerance
(opening at half the coarsest mesh's spacing), a mark and a line for
each move, a click to leave one out, and an *Align* button. A move that
would turn an element inside out is refused before the button is
pressed. The nodes moved are drawn as blue squares, ready for *Merge
Coincident Nodes*. From a script, `project.align_meshes(name, reach,
groups=[...], exclude=[...])` (`Geometry.align_moves` lists the moves,
each a node and the node it moves onto).

*Mate Surfaces* is for parts whose meshes meet face to face, matched
or not. Pick two element groups or more in the tree, plates or solids,
and it is on the bar. Its pane finds every patch where one group's
outside lies on another's and faces it, within the tolerance, and
lists them, one entry per connected patch, so four legs standing on a
table top are four entries. The first side of a patch is the finer
mesh; its nodes are tied to the second side's surface.

To change a patch, pick it and pick the side with the two buttons.
Only that side's nodes are drawn, so coincident nodes of the two sides
are never confused: the patch's nodes large, the side's nodes in the
other patches small, and its other nodes that can be picked small in
a muted color. A node the two groups share, where the mesher or a
merge made the meshes one node, is a square on either side: it is
joined already, tighter than any tie, and is never tied or picked. A
node can be picked only within the *Pick within*
distance of the other side's surface, and the node under the cursor
lights up. A click adds the node to the side or takes it away.

How the sides are held is set twice, once along the surface's normal
and once in its plane (the shear): *Rigid*, *Stiffness* (per node, in
the display units) or *Free*. Rigid in both ties the surfaces together;
rigid along the normal and free in the plane lets them slide; a
stiffness in both is a joint to tune until the model's modes match the
measured ones. Each node of the first side is tied to the point of the
second side's surface it lies on, moving with the corners around that
point, and along the surface's normal there. On a curved surface, a
shaft in its bore, each node has its own normal, so a shaft held
rigidly along the normal and free in the plane turns and slides in its
bore and nothing else. When the model is built, each tied node is put
on the point it is tied to, which it is already within the tolerance.
*Keep* adds the picked patches to the geometry's **Mated Surfaces**
row, one set each, named *post on base* or as typed, and the pane stays
open on the patches left.

A set of mated surfaces holds no elements. Its ties are made when a
model is built, so the set can be changed at any time. Its row's pencil
opens the pane on it, to change its nodes, how it is held or its name,
and *Save*. *Delete* removes a set, or every set when the **Mated
Surfaces** row itself is picked. A set follows its nodes when they are
renumbered or merged, and goes with its element group. To make the
meshes one where their nodes coincide, use *Merge Coincident Nodes*
instead. From a script, `project.mate_surfaces(name, ['post', 'base'])`
keeps every patch found, rigid in both unless `normal_mode` and
`shear_mode` say otherwise; `project.detect_mated_surfaces` lists the
patches, and `project.add_mated_surfaces`, `edit_mated_surfaces` and
`delete_mated_surfaces` work on one set, stiffness in N/m.

### Solids

Solids are the shapes a part is built from before it is meshed. They
are listed on a geometry's **Solids** row, which is there before the
first solid is added. *Add Block*, *Add Cylinder*, *Add Cone*
and *Add Sphere* on that row open the add pane on a solid centered on
the origin. A block is a center, three widths, three angles and an
optional corner radius, which rounds its four edges along its own Z so
that a window cut with it has fillets. A cylinder is a center, a
radius, a length along its own Z and three angles. A cone is a frustum:
a center, a base radius at its own -Z end, a top radius at its +Z end,
a height and three angles, with either radius zero for a point. A
sphere is a center and a radius. Each takes a name. The pane draws the
shape as you type, and the rings and arrows turn and slide it. *Add
Section* adds an I-beam, a channel, a tee, an angle, a rectangular tube
or a round tube, run a length along its own Z and sized by the same
dimensions the beam element's sections take (a tee takes an I-beam's).
Each of these is one solid, and its pencil changes its dimensions.

A hole is a cylinder subtracted from a part, or two cylinders for a
counterbore. Projects made with the hole solid that was once on this
row still open with their holes, which draw, mesh and edit as before:
a hole's pencil sets its depth, or *Through*, and its *Counterbore*.

Pick two parts or more in the tree and the bar offers *Union*,
*Subtract* and *Intersect*. *Subtract* takes the others from the one
first in the tree, which its tooltip names, so a plate with holes is
the plate added first, its holes after, all of them picked, and
*Subtract*. Drag a solid's row up or down among the rows beside it to
change the order: the parts among the parts before a boolean is made,
or a boolean's own solids under its row afterward, which changes what an
existing subtraction cuts. A blue line shows where the row will land;
a row only moves among its own siblings. A boolean lists the solids it
is made of under its own row; one picked there is drawn over it, so a
hole's cylinder shows inside the plate. *Pattern*, on picked solids, makes copies of them in rows,
columns and layers: how many along X, Y and Z and how far apart, up to
10,000 copies. The result is one solid with the originals under it,
and the copies follow any change to the originals. A pattern can be
patterned again. *Take Apart* undoes a boolean or a pattern. Delete
removes a part whole, with the solids inside it; a solid that a boolean
still uses cannot be deleted on its own.

The pencil on a solid's row opens the same pane on it, to change it
after it is made: type a block's or a cylinder's numbers or drag the
rings and arrows, and the change is drawn as you make it; *Apply* makes
it. A hole changed this way is changed in the part it was subtracted
from, and a patterned hole in every copy. A pattern's pencil opens
*Pattern* on its counts and spacing. A combined part has no size of its
own, so its pencil moves and turns it whole, with every solid it is
made of. A mesh already made from the solids stays as it was until
*Mesh Solids* meshes it again.

The parts are drawn see-through in their own color, and the picked
ones in the selection color. A solid is not an element, and nothing
solves or exports it until it is meshed. Solids are evaluated by the
OpenCASCADE kernel, which is installed with the package. From a script, use
`project.add_solid_block`, `add_solid_cylinder`, `add_solid_section`,
`add_solid_hole`, `combine_solids`, `pattern_solids`,
`separate_solids`, `delete_solids`, `edit_solid` and `move_solids`; the
shapes themselves, their volumes and their surfaces are in
[`core.solids`](../api/visualdynamics.core.solids.md).

*Add Plate* and *Add Disk* add flat shapes. A plate is a center, two
widths in its own X-Y plane and three angles, and a disk is a center, a
radius and three angles. Neither has a thickness: they are meshed into
shells, and the thickness is typed for their element group in the
Element Groups table. A block, a cylinder, a section or a hole
subtracted from a plate cuts it where it crosses the plate's plane, so
a plate less some cylinders is a plate with holes. A union is of flat
shapes or of solid ones, not both, since the two mesh into different
elements.

### Mesh Solids

*Mesh Solids*, on the bar when parts are picked, meshes them into
eight-node bricks, ten-node tetrahedra or shells, each part an element
group named for it. The element type is yours to choose in the pane:
*Hexes* for the parts that are an extrusion, *Tetras* for any shape,
and *Shells (quads)* or *Shells (triangles)* for plates.

A plate or a disk is meshed into shells on its own plane, with its
holes round, the same way bricks are rounded; with a plate or a disk
among the parts, only the two shell types are offered. A solid with a
volume meshed into shells is meshed on its outer surface only: the
quads are the outside faces of its brick mesh, and the triangles the
outside faces of its tetrahedral mesh. Triangles on a plate are its
quads, each split on its shorter diagonal. Shells take a material and
a thickness in the Element Groups table before the model is solved.

Bricks mesh an extrusion best: every block square to the geometry's
axes, and every cylinder and rounded corner along one axis. That
covers plates, bars and their holes. A cylinder along a second axis, a
hole drilled in from the side, is cut from the grid and its nodes
pulled round in its own plane, without a ring, so the bricks around it
are uneven. The grid is cut at every flat face of every block, so flat
faces are meshed exactly, and each span is divided into the whole
number of bricks nearest the element size. A brick belongs to the
picked part whose shape holds its center, and where parts overlap it
goes to the one later in the tree.

A union that is not an extrusion is meshed in pieces, one per
operand, as deep as unions go: a union with another union among its
operands gives up that union's operands too, so a table unioned again
with a sphere is still its block, legs and rod, and the sphere. A
subtraction whose first operand is a union is split the same way, each
member less what the subtraction takes: a block and a flange unioned
and drilled through are the block less the hole and the flange less the
hole. Any other subtraction, an intersection or a pattern is one piece,
its operands being no material on their own. Each piece that is an
extrusion is swept as above, with rings about its holes, and each that
is not is fitted as below, every piece at its own size. A piece that is one block, cylinder or section turned
off the axes is swept along its own axis: a tilted rod gets rings along
its length. Where it runs into a piece before it, each line of nodes
along the sweep stops at that piece's surface and its nodes are spread
evenly back to the free end, so the last layer lies on the face it
meets and the bricks near there are sheared; the pane gives the worst
brick. A turned piece that runs through another, or meets pieces at
both ends, is fitted instead. One turned block, cylinder or section
meshed on its own is swept the same way. The pane then shows a size per piece in place of the
one box, each starting at the size the piece is likely to hold at,
and *Detect Element Sizes* fills every row. Where two pieces meet and
their grids match, the seam's nodes are shared, and where they do
not, the finer piece's nodes on the seam are tied rigidly to the
coarser piece as mated surfaces, one set per seam under the geometry's
**Mated Surfaces**, which can be given a stiffness afterward; the pane
reads both counts. Where two pieces overlap, the material is the
earlier operand's: a later piece's grid is cut at the earlier pieces'
faces and its bricks inside them are left out, so legs that run up
into a table top end at its underside. Each piece keeps its own
element group, named for its operand. This is how a part with one thin piece on a thick one
meshes: a block with 0.2 in flanges wants 0.2 in bricks on the block
and 0.067 in on the flanges, which no one grid can give. In a script,
`mesh_solids(name, [part], 0.2, unit='in', sizes={flange: 0.067})`
names the sizes by solid id.

A shape that is not an extrusion, a hole drilled at an angle, a cone or a sphere, is still
meshed into bricks, on a grid fitted to it: the cubes whose centers
lie inside are kept, a layer of bricks is laid under their skin, the
skin's nodes are moved onto the shape's surface with its corners and
sharp edges captured, and every node is then moved to where the worst
brick around it is best. The pane says so (*on a grid fitted to the
shape*) and reads the worst scaled Jacobian (1 for a cube, 0 for a
flat brick), which comes out between about 0.15 and 0.3 where a ring
mesh gives 0.5 or better. The fitting takes seconds to minutes, so the
preview draws it only for a small grid. A feature only a few bricks
across leaves no room to fit: a fit has held with five or more bricks
across the smallest round feature's radius on most shapes measured
and folded with four or fewer on all of them; where a bore leaves a
surface at a shallow angle, the feather edge there can fold a few
bricks at any size. *Detect Element Size* writes the five-across size
into the box (the smallest radius over five, or the
thinnest wall over three; for an extrusion, half the thinnest
feature; for tetrahedra, the radius over three), and *Mesh* on a size
under four and a half across, or under two and a half through the
thinnest wall, asks first, with the size likely to hold as the
default, before the minutes are spent. Where the size likely to hold
would itself be a grid past the brick limit, the part wants a finer
size on its thin piece than on the rest, which one fit cannot give,
and the question offers tetrahedra instead. A fit that folds anyway
is refused with the count, and the pane says to use a smaller size or
*Tetras*.

Holes, bosses and inserts are meshed round. About each circle with
room around it the grid is replaced by a ring mesh, with the circle's
nodes evenly spaced on it, layers of bricks out to the grid, and a
square core in the middle where the circle is filled. Circles about one
center, such as a washer about a hole, make one ring. The ring's
outline is a rectangle of the grid's cells where one fits, and steps
around whatever is in the way where none does (a hole nearly as wide as
a table's top, beside its legs). Where there is no room (a hole close
to another hole or to a face), and at a rounded corner, the grid's
nodes on the rim are pulled onto the circle instead; a pull that would
turn a brick inside out is made as far as it can be.

Tets mesh any shape the solids describe, a hole at an angle included,
from the shape alone. Points are spread through the part and relaxed
until the edges between them are even, points pushed out of the part
are put back on its surface, and the sharp edges and corners of the
part are held by points placed along them, so nothing is rounded off.
The elements are ten-node tetrahedra, their midside nodes on the
curved walls, which the solver takes alongside bricks; the four-node
tetrahedron is far too stiff in bending and is not offered. The pane
reads the count, the worst radius ratio (1 for a regular tetrahedron,
0 for a flat one) and how many are under 0.3. A tetrahedral mesh
takes longer than a brick mesh of the same size, seconds for a
hundred thousand elements, and the preview draws it only up to a few
thousand; past that it says the count and *Mesh* makes them with the
progress bar up. Where both kinds can mesh a part, bricks are the
better element for the same count, and a brick mesh of a plate or a
bar should be preferred.

Parts meshed together share nodes where they touch (a frame and the
inserts in its holes). *Keep apart from other groups* lands the bricks
only on their own groups' nodes. With two
parts or more picked, *Keep the parts apart from each other* meshes
them on one grid with each part keeping its own nodes where it touches
another, so parts to be bolted lie node on node, ready for *Merge
Coincident Nodes* (the BARC's box, channels and beam are meshed this
way). The pane opens on an element size of half the thinnest feature,
draws the bricks as the size is typed, and reports how many there are
and the shortest edge; a size that would make more than 2,000,000
bricks is refused. The count is the parts' volume over a brick's, so a
part filling a small share of its box is held to the bricks it will
have, not to its box. The grid a mesher lays over the box is held to
10,000,000 cells, which is about 3 GB of memory while it is laid. *Mesh* adds them and closes the pane. The solids
stay, to be meshed again or deleted. When a part is meshed again it
replaces its earlier mesh: the bricks of the element group named for it
are removed, with the nodes only they used (unless data is measured or
solved there), and the group keeps its material. The pane says how
many bricks will be replaced before you press *Mesh*. From a script,
`project.mesh_solids(name, ids, size, unit=..., apart=...,
parts_apart=...)`.

## Photos — [`core.photos`](../api/visualdynamics.core.photos.md)

Kind `photos`. Setup photographs as they arrived (`names`, `formats`
and the encoded `images`), never re-encoded: a report embeds them, and
a second JPEG generation would only lose quality. Methods: `add_file`,
`rename`, `delete_photos`, `plot`.

## Channel table — [`core.channel_table`](../api/visualdynamics.core.channel_table.md)

Kind `channel_table`. One `frame`, with a row per channel and a fixed
set of typed columns. It comes from a controller run, a spreadsheet or
a project file. Methods: `dof_strings`, `roles`, `controls`, `types`,
`sensitivities`, `ranges`, `units_for`, `set_cell`, `rename_dof`,
`delete_channels`, `save`. [The channel table](channel-table.md) has
its rules.

## The data arrays — [`core.data`](../api/visualdynamics.core.data.md)

Every measured or computed curve is a `DataArray` subclass, and one
object holds many records sharing one abscissa: `abscissa` (the x
axis, in seconds or hertz), `ordinate` shaped `(records, samples)`, and
one entry per record of `response_dof`, `reference_dof` where the type
has a reference, `block` (which repeat), `ordinate_dim` and
`ordinate_unit` (the quantity and its SI unit), `reference_unit` for a
ratio, `dimension_hint` (what a file claimed without giving its scale)
and `comment`. Uneven and unsorted abscissas are accepted on import;
anything that needs an even step checks for it when it is used.

Every data array has `plot`, `plot_waterfall`, `save_plot` and `save`;
`define_units` and `undefine_units`; `delete_records` and `rename_dof`;
`display_ordinate` and `display_abscissa` in a chosen unit system; and
`num_records` and `record_label`.

The rows of a data object's grid in the project tree are its
coordinates, and the columns of a matrix are coordinates too. A channel
assigned to the wrong point at the instrument is corrected in the grid:
double-click the row or column label and type the coordinate. The
change applies to that channel, meaning that coordinate and the
quantity of that row or column, wherever it appears in the object. A
CPSD's accelerometer is on both sides of its cross terms and moves as
one sensor, while an FRF's drive-point accelerometer (a row) and load
cell (a column) are two channels, and moving one leaves the other to be
moved separately. Everything derived from the object follows, because
a PSD computed from a mislabeled channel is mislabeled the same way
(`project.rename_dof`; without a quantity it moves every channel at the
point). A rename that would give two records the same coordinate and
quantity is refused: a load cell and an accelerometer can share a
point, two accelerometers cannot.

A channel table's row coordinate is its node and direction. Type over
the row in the grid and the two cells change; type a node or a
direction in the table view and the row's coordinate follows. A channel
table imported with the data it describes is linked with it (one file,
one group), but it is otherwise its own object: a coordinate corrected
on the time history is not corrected on the table, or the reverse.

| kind | class | what it is | what it adds |
| --- | --- | --- | --- |
| `time_history` | `TimeHistory` | the record as acquired, against time; carries its `averaging`, `shocks`, `roles`, `truncation` and `filtering` settings so every result computed from it reads one description | `compute_spectra`, `compute_psds`, `compute_cpsds`, `compute_frfs`, `compute_multiple_coherence`, `compute_srs`, `integrate`, `differentiate`, `filter`, `truncate`, `animate`, `split_into_frames`, `sample_rate`, `drive_dofs` |
| `transient_specification` | `TransientSpecification` | a target waveform: what a transient test was controlled to, sample by sample | everything a time history has |
| `spectrum` | `Spectrum` | the complex average of a record's frames, amplitude and phase per line | `animate` |
| `psd` | `Psd` | the power average: real autospectra, complex cross spectra; `bin_widths` when the bins are not even | `area`, `to_octave`, `bin_bounds`, `coherence`, `principal_shapes`, `as_specification`, `animate` |
| `specification` | `Specification` | a PSD with its band: the target and the warning and abort limits on either side, as arrays beside the ordinate | `limit`, `has_limits`, `as_psd` |
| `frf` | `Frf` | response per unit reference, complex, one reference per record | `plot_cmif`, `animate`; the input to a modal fit |
| `coherence` | `Coherence`, `MultipleCoherence` | how much of a response one reference explains, or all of them together; bounded 0 to 1 | `plot_map` |
| `srs` | `Srs` | a shock response spectrum: the peak an oscillator of each natural frequency reached, laid out in octaves | `damping` |
| `shock_specification` | `ShockSpecification` | an SRS with its band, conventionally +6 dB and −3 dB | `limit`, `has_limits` |

### Writing and editing a specification

A specification can also be written in the app, on a sheet that opens
two ways. *Create Specification* on the table bar of a single channel table
starts one at the table's control channels: the channels with Control
checked, or every motion channel when none is. With rows picked in the
table's grid it starts at those rows instead, whatever their role, so a
force CPSD is written at the reference channels. *Edit* on the plot bar of a
specification opens the specification itself, with all of its channels
or only the ones whose records are picked in the tree. Several
specifications opened at once share one sheet, with a column per
channel under its specification's name and cross terms within each.
Opened on a channel table, the sheet first makes the specification
beside it, in the table's object group, and edits that.

A specification at a shape set's modal coordinates, such as a virtual
point's target, is written from a script:
`core.author.SpecificationDraft.at_modal_coordinates(shapes, modes=...)` starts the
sheet, using every shape or only the ones named (a virtual point's
three translations, with the rotations left out), and
`project.author_specification(name, draft)` makes it.

The sheet sits beside the plot. It holds the breakpoints and a level
per channel, each pair's coherence and phase (a pair left unstated is
absent, and independence is a statement too), *Add breakpoint* and
*Remove breakpoint*, *Scale selected* with a box that raises or lowers
the selected levels by a number of decibels, and an *All pairs* row
that states one coherence and phase for every pair. *Enter: Levels* or
*Slopes* chooses how the levels are typed: a level at every breakpoint,
or a level at the first and a slope in decibels per octave into each
one after it, the level there computed. *Areas and running rms* writes
each segment's area on the plot, draws the rms accumulated from the
low end under it, and reads both out under the breakpoints; both
are offered on the breakpoints form
([Writing a random specification by slope](breakpoint-slopes.md)). While the sheet is
open the plot is the 2-D one, since every editing gesture happens
there, and the 3-D stage is off until the sheet closes. Selecting
another object in the tree closes it, the channel table it was made
from included; coming back to the specification shows it drawn, with
*Edit* to open the sheet again.

The warning and abort bands are not on the sheet: they are on the
plot, as the shaded zones. While the sheet is open each edge of each
band has a drag handle. There is one handle over the whole range while
the channel is *Uniform*; with *Uniform* off there is one per linear
section of the requirement (each segment between breakpoints, or each
run of one power law in the interpolated form), and each is moved on
its own. Drag one up or down; it snaps to quarter decibels and its
label shows the value. When you let go, that edge moves to that value
on every channel the sheet holds in that frequency range, which is the
whole specification or the channels picked in the tree. A
specification with no limits opens with the defaults, ±3 dB warning
and ±6 dB abort, shown on the plot to drag; the sheet says they are not
on the object yet, and the first edit writes them. Two constraints,
after *Bands:* under the form buttons, apply to every channel of the
specification the same way: *Symmetric* keeps the band above the
target equal to the band below, and *Uniform* keeps every segment at
one band. Each box opens checked when every channel's own bands meet
that constraint.

Both grids are the application's ordinary tables. A column header
selects a channel, a row header a breakpoint, and Cmd-click adds cells;
copy, paste, Delete to clear, Batch Edit and the fill handle work as
they do everywhere else. A frequency typed between two others re-sorts
the breakpoints. There is no Apply button: every edit is written to the
specification as it is made, so switching to another object never
leaves an edit behind.

A sheet holding every channel rewrites the object in the sheet's own
form: fold a controller's target to its breakpoints and the object
becomes the breakpoints. A sheet holding a picked subset of channels
merges back at the object's own lines and leaves the other channels
unchanged, so the object's name, links and place in every report stay
the same (`project.author_specification` with a
[`core.author.SpecificationDraft`](../api/visualdynamics.core.author.md);
`replace=True`, and a list of names for a sheet covering several). A
pair left unstated is absent from the specification, not assumed. A
cross term that varies with frequency opens unstated, and the sheet
says so.

A sheet has two forms, chosen by the pair of buttons at its top and
applied to the whole specification whichever channels the sheet is
open on. *Breakpoints* is the few points a requirement is written from.
*Interpolated* is the same requirement read onto evenly spaced
frequency lines, the way a controller writes its target. A controller's
target opens as lines and folds to its breakpoints exactly, since its
lines were computed from them, and editing is done on the few rows. A
specification with no power-law structure keeps every line and says
so. The line spacing is the specification's own, otherwise a time
history's averaging (sample rate over frame length), otherwise the
value typed into the box beside the buttons; nothing is guessed. A
specification written at modal coordinates and expanded through the
set is an exact control-channel target with every cross term.

*Save As* offers a specification as *Rattlesnake random specification
(.npz)*, the target file the controller's Random environment loads
before a test. It holds the frequency lines, the whole cross-spectral
matrix at each line, the four band edges, and, when every channel is a
node and a direction, the node and direction of each channel so the
controller can put the matrix in its own channel order. Most random
tests run on autospectra alone, and the controller's form of that is a
matrix with zeros off the diagonal. So a pair the specification does
not hold is written as a zero, and a `held` mask records that it was a
placeholder rather than a statement of independence. The values are
written in the coherent unit system on display; the file cannot record
units, so the status line names the system. The same file imports
again, with held cross terms kept and placeholders left out. A target
file written by the controller or by sdynpy imports too, with an
off-diagonal that is all zero or NaN read as the placeholder for
autospectra alone.

## Sine — [`core.sine`](../api/visualdynamics.core.sine.md)

Two kinds that are not data arrays, because each tone sweeps its own
frequencies on its own clock, and different abscissas cannot share one
array. `sine_sweep_specification` (`SineSweepSpecification`) is what a
sine test was controlled to: the `tones`, each with its breakpoints,
sweep law, bands and start time, over the control DOFs. `sine_levels`
(`SineLevelSet`) is one extraction: the per-tone `levels` a joint
Vold-Kalman solve read out of a recording, grouped the way the
specification groups its tones. Each level carries the noise floor
beside every value (`floor`) and where the tone was below it
(`below_floor`, the value then reported at the floor). The set carries
the smoothing it was read with (`cycles`), and a level whose sweep
clock was corrected says by how much (`drift_hz`).

## Shapes — [`core.shapes`](../api/visualdynamics.core.shapes.md)

Kind `shapes`. Mode shapes over a shared set of DOFs: `frequency` and
`damping` per mode, `shape_matrix` shaped `(modes, dofs)` with
`coordinate` naming each column, `modal_mass`, `modal_damping` and
`mass_unit` where a source carried them, a `description` per mode, and
`unscaled` when the fit had no drive point to set the scale. A fitted
set is also the record of its fit, which is what *Edit Fit* reopens. A
geometry's rigid-body set (three translations and three rotations about
its reference point, with `frequency` exactly zero) is also `unscaled`
unless mass and inertia were given, in which case it is mass-normalized
about the principal axes of the inertia.

Any shape set can carry data through itself. `project.transform` fits a
record's motions to the modes (`q = Φ⁺u`) and projects its forces
(`Φᵀf`), giving one record per mode and quantity at the modal
coordinates `M1` … `Mn`. The name is the letter M and the mode's index,
so it cannot be mistaken for a node, and it is the same for every set;
the object's provenance says which set it came through.
`project.expand` carries modal responses back to every DOF the set
covers. Each kind of data is transformed in its own form. A time
history or spectrum is transformed row by row. A CPSD or a
specification is transformed as the matrix, `S_qq = Φ⁺ S_uu Φ⁺ᴴ`,
which needs every cross term between the shared channels. Autospectra
alone are refused rather than filled in with a guess, so compute the
CPSDs, or import the specification with the cross terms the controller
wrote. The tolerance bands are carried exactly when every channel has
the same one. An FRF is transformed on its response rows and, when the
set covers the drives, on its reference columns too. A shock response
spectrum or a coherence does not transform, since one is a maximum and
the other a ratio: transform the time history and compute it again.

The expansion uses every mode for the motion of the structure, or only
the modes picked in the tree for their contribution alone, `u = φₖqₖ`,
and the result is named for them (`records=` in a script). Both results
go into the group with the set and the record. A modal object is attached
to the shape set it came through, not to a geometry, and works with
any set that has every mode it names. Both carry the record's
averaging frames and shock windows, which fall on the same instants.
The unit rule is `[q] = [u]/[Φ]`: unit rigid shapes give the virtual
point's rotations in rad/s² and its moments in lbf·in, mass-normalized
shapes give the half-power mass units (`modal_acceleration`,
in/s²·slinch½), and a set with no mass unit gives responses whose units
are left to declare. In the window, select the record and the set
together and press *Transform to Modal Coordinates* on the bar (or
*Expand to Physical Responses*, for a modal record such as the
[modal responses computed from forces](modal-responses.md)). The object is made
at once, since a transform has no settings, and the status line reports
the DOFs shared, the DOFs left out and the residual the fit leaves. The
same report is kept on the object as its `transform_report`
([`core.transform`](../api/visualdynamics.core.transform.md)). The
matrix itself, with its conditioning and the residual it leaves against
frequency, is a `Transformation` ([The transformation to a
point](transformation.md)).

Methods: `plot`, `animate`, `animate_pair`, `plot_mac`, `auto_mac`,
`synthesize_frf`, `covers`, `delete_modes`, `save`.

## Matched modes — [`core.matches`](../api/visualdynamics.core.matches.md)

Kind `matches`. Pairs committed on a cross-MAC between two shape sets:
`first` and `second` name the sets, `pairs` the modes, and `macs` the
value each pair had when it was committed. Methods: `add`,
`delete_matches`.

## Report — [`core.report`](../api/visualdynamics.core.report.md)

Kind `report`. An ordered list of blocks, each a plain dict, bound to
project objects symbolically by default (`@basis:Frf`) or by a literal
name. The blocks are text, plot, scene, table, photo, bars, the
pass/fail box (`verdict`), and the matched-modes pairs and their
overlay. Methods: `add`, `remove`, `move`, `unbound`.
[Reports](reports.md) explains the bindings.

## What can be done with what

The processing verbs are on the project, so a result is added, named
and linked to its source in one call:

| on | verbs |
| --- | --- |
| a time history | `filter_data`, `truncate_data`, `detect_shocks`, `compute_spectra`, `compute_psds`, `compute_cpsds`, `compute_srs`; `compute_frfs` and `compute_multiple_coherence` when it has both reference and response channels; `extract_sine` when the project holds a sine sweep specification; `integrate` and `differentiate` when the quantity allows; `transform` through a shape set whose DOFs it is measured on, and `expand` back when it holds that set's modal responses |
| a geometry | `generate_rigid_body_modes`; `set_view`; `add_solid_block`, `add_solid_cylinder`, `add_solid_cone`, `add_solid_sphere`, `add_solid_section`, `add_solid_hole`, `add_solid_plate` and `add_solid_disk`, then `combine_solids`, `pattern_solids` and `mesh_solids`; `tie_elements`, a patch of elements tied rigidly to an element group or a second patch; `merge_coincident_nodes` and `align_meshes` once it has elements, for the whole geometry or the element groups named by `groups=`; `merge_groups`, element groups of one element type and the same properties made one; `solve_modes` once its element groups carry their properties (a material and a thickness or a section per group in the Element Groups table, a material alone for a group of solids), which builds the finite element model and solves it |
| the project itself | `new_geometry`, an empty geometry to build in; `new_channel_table`, an empty channel table to fill in; `generate_report` |
| a channel table | `add_channels`, blank channels numbered on from its highest |
| a PSD, CPSD or specification | `compute_octave`, with a specification's warning and abort limits banded along with it; `convert_to_specification` for a PSD that is a requirement, and `convert_to_psd` for a specification, its limits left out |
| a shape set, specification or channel table | `author_specification` |
| an FRF | `fit_modes` |
| two shape sets | `project_onto_basis`, `match_modes` |
| a shape set and a channel table on another geometry | `project_onto_basis`, the modes at the table's channels |
| any object with a sibling of its type | `merge` |

Every object also works with the whole-project verbs (`add`, `remove`,
`rename`, `duplicate`, `link`, `unlink`, `place`, `set_basis`,
`export`, `save`, `generate_report`, `export_report`, `refresh`), and
`project.verbs()` with no argument lists all of the processing verbs
with their descriptions. The API page for
[`project`](../api/visualdynamics.project.md) has every signature.
