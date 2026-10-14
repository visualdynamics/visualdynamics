# The modal workflow

This page takes a modal test from import to identified modes and a
report, in the app, in a script, or partly in each. The window holds
the same `Project` a script builds and calls the same methods, so a
project built by clicking and one built by running code are the same
project, and each opens in the other.

The walkthrough is one complete survey, with a screenshot at each
step: the demonstration plate's modal test, a burst-random Rattlesnake
run on two shakers. The same files ship with the repository
(`testdata/plate/`), so every step can be repeated exactly.

The table lists the steps first; each is described in full below it.

| Step | In the app | In a script |
|---|---|---|
| Start a project | it starts empty | `project = visualdynamics.Project('Name')` |
| Import | drag files anywhere on the window, or File → Import... | `project.import_file(path)` |
| Declare units | the Imported Units pane | `obj.define_units('m')` |
| Group what belongs together | select, then **Link** on the bar above the tree | `project.link('Geometry', 'FRF', ...)` |
| Name the measured side | right-click the bracket → Basis of Comparisons | `project.set_basis(...)` |
| PSDs from the captures | **Compute PSDs** in the Averaging view | `project.compute_psds('Time History')` |
| Identify modes | the fitting screen: **Find Mode**, drag, **Confirm Mode** | `project.fit_modes('FRF', bounds=(300, 1300), limit=5)` |
| Correlate | select both shape sets and the MAC appears | `project.comparison_mac('A', 'B')`; `project.plot_mac('A', 'B')` draws it |
| Commit matches | pick squares, press **Add Matches** (+) | `project.match_modes('A', 'B', threshold=0.7)` |
| Report | *Generate Report* on the bar, the project row selected | `project.generate_report('modal')` |
| Or all of it at once | **Automatic** on the bar above the tree, once the project has its type | `project.work_up()` |
| Export | Export → HTML... on the report's bar | `project.export_report(name, path)` |
| Save | File → Save Project As... | `project.save('modal.vdyn')` |

## In the app, step by step

### 1. Import the survey

1. Drag the controller's spectral save (`modal_spectra.nc4`) and the
   test geometry (`test_geometry.npz`) anywhere onto the window.
2. The file says what kind of run it was, so the project type changes
   to *Modal Test* on its own. The tree then shows gray slots for
   everything a modal report draws from. Each slot disappears when the
   object it stands for arrives, so the gray slots that remain are
   the steps still to do.

![The window after the import: the tree holding the measured objects
with the type's remaining slots in gray, and the FRF's 22 records
drawn, with the plate's four modes standing clear and the survey's one
in-plane channel about eight orders of magnitude lower, which is what
a channel pointed where nothing moves reads](../images/modal-import.png)

### 2. Declare what the files did not say

The controller's file carries its units, so its objects arrive with
units defined. The geometry came from an `.npz` that does not record
a unit, so it arrives without one.

1. Select the geometry.
2. In the *Imported Units* pane, enter the unit for its one row
   (Coordinates, length): meters.

Nothing is ever scaled by a guess: an object with undefined units
shows a badge until you give it one.

The article's CAD can be shown with the survey too: a STEP, IGES, 3MF
or STL file imports as face elements, one element group per part, and
can be linked into the same object group for context.

![The Imported Units pane for the geometry: one row, Coordinates /
length, the Unit cell waiting for
meters](../images/modal-units.png)

### 3. Link the measured side, and make it the Basis

1. Select everything the test measured and press **Link** on the bar
   above the tree. The bracket drawn down the tree's left edge marks
   the group.
2. Right-click the bracket and check *Basis of Comparisons*.

Comparisons are made in the Basis group's DOFs, it is the reference
for frequency error in the matched-modes table, and every `@basis:`
binding in the report resolves through it.

![The measured group bracketed as one side, with the FRF
selected](../images/modal-link.png)

### 4. PSDs from the captures

The time data is twenty burst captures, one per accepted average.

1. Select the time history and turn on **Averaging** on the plot bar.
   The averaging view plays the captures one at a time, the pager
   steps between them, and the panel shows how many averages pool
   (*20 × 1 = 20*).
2. Press **Compute PSDs**. The densities are averaged over exactly
   those frames, the run's own averaging, and are added beside the
   record, linked into the group.

![The computed PSDs, one record per channel, averaged over the run's
twenty captures](../images/modal-psds.png)

### 5. Identify modes on the fitting screen

1. Select the FRF and choose *Fit Modal Model* on the bar. The
   fitting screen opens with the CMIF on top, the mode table below on
   the left and the MAC on the right.
2. Press **Find Mode**. The cursor moves to the most prominent CMIF
   peak left in the residual, within the frequencies on the plot
   (zoom to limit the search).
3. Drag the cursor if the pole needs adjusting. The fit follows the
   cursor and settles when it stops. Check the dashed synthesis
   against the measured CMIF, and the MAC.
4. Press **Confirm Mode**. The mode is added to the table and the
   cursor moves to the next peak.
5. Repeat steps 2 to 4 for each mode. Once two modes are confirmed,
   **Refine All** appears at the left of the bar: it refits every
   mode's residues jointly with all the modes present, for close modes
   that affect each other (`fit_modes(..., refine=n)` in a script).

![The fitting screen as it opens: the measured CMIF with the first
suggestion already on the largest peak](../images/modal-fit-open.png)

The cursor is two controls in one: moving it left and right sets the
frequency, and dragging it vertically holds the damping. The dashed
parabola under the cursor shows the mode about to be fitted, and its
width shows the damping. The algorithm, and how it differs from the
batch polynomial methods, is described in
[How the mode fitting works](../modal-fitting.md).

![Mid-fit: two modes confirmed and bookmarked, the cursor's parabola
on the 647 Hz peak at 1.98% damping, the confirmed modes in the table
with the auto-MAC beside them](../images/modal-fit.png)

Five confirms give the plate's five modes in the band: four distinct
peaks, and the second mode of the repeated pair at 1142 Hz. Find Mode
offers that second mode at the frequency already confirmed because the
shape left in the residual there is not the confirmed mode's shape.

Each confirm publishes the fit as a shape set, and that shape set is
the complete fit state. To change the fit later, select the shape set
together with the FRF and choose **Edit Fit** on the plot bar; the
session reopens with every mode as it was published.

### 6. Look at a shape

1. Select the fitted shape set together with its geometry.
2. The modes animate on the survey geometry, deflected and colored by
   displacement.

![The first identified mode, the plate's twist at 439 Hz, drawn on
the survey geometry with the corners moving and the center
still](../images/modal-mode.png)

### 7. Correlate against the model

1. Import the FEM pair, the full meshed plate and its computed modes,
   and declare their units.
2. Link the two as their own group. The pair can come from the
   solver's own files: a Nastran bulk data file (`.bdf`) for the mesh
   with a punch file (`.pch`) for the eigenvectors, a Femap Neutral
   file (`.neu`) carrying both, or an Exodus mesh with sdynpy or UNV
   shapes. The [interchange matrix](../../export.md) has the full
   list.
3. Select the two shape sets together. The cross-MAC appears, with
   the set that has more modes down the side (here the FEM modes) and
   the other across.
4. Click a square to pick that pair; the status bar reports the two
   modes and their MAC. Shift-click to add more squares to the
   selection.
5. Press **Add Matches** (+) to add the selected pairs to a Matched
   Modes object for the two sets, created the first time.

![The cross-MAC: five fitted modes against the model's twenty-two,
the five matches along the diagonal of the band, the picked pair
outlined and described in the status bar](../images/modal-mac.png)

### 8. Generate the report

1. Select the project row and press *Generate Report* on the bar (or
   use File → Generate Report → Modal Test). Every figure and table in
   the modal report binds symbolically (`@basis:Frf`, not a name), so
   renaming objects later does not break it.
2. Fill in the text blocks, such as the test summary and the
   conclusions.
3. Choose Export → HTML... on the report's bar. It writes one
   self-contained file that opens in any browser.

![The exported report: the title page, test geometry and
instrumentation figures, generated from the
project](../images/modal-report.png)

## The same survey, headless

You do not have to write the script by hand. The *Console* tab along
the bottom of the window records the session as you work, each action
as the line that replays it. Expand it, copy the part you want, and it
runs as it stands.

The same sequence as a script is below. It is the sequence in
[`examples/modal_workflow.py`](https://github.com/visualdynamics/visualdynamics/blob/main/examples/modal_workflow.py),
which the test suite runs, with the names this page uses. The step
numbers match the walkthrough above.

```python
import visualdynamics

project = visualdynamics.Project('Plate Modal Survey')
project.project_type = 'Modal Test'

# 1. import — the run switches the type and fills the first slots
project.import_file('modal_spectra.nc4')
project.import_file('test_geometry.npz')

# 2. declare what the files did not say
project.geometry.define_units('m')

# 3. the measured side is the basis
project.set_basis(*project.names)

# 4. PSDs over the run's own captures
project.compute_psds(project.basis.time_history)

# 5. identify the five modes in the band
project.fit_modes(project.basis.frf, bounds=(300.0, 1300.0),
                  limit=5, name='Identified Modes')

# 7. the FEM pair, and the matches
project.add('FEM Geometry', visualdynamics.import_file('geometry.npz'))
project.add('FEM Shapes', visualdynamics.import_file('shapes.npy'))
project.link('FEM Geometry', 'FEM Shapes')
project.other.geometry.define_units('m')
project.other.shapes.define_units('kg')
project.match_modes(project.basis.shapes, project.other.shapes,
                    threshold=0.7)

# 8. report, export, save
project.generate_report('modal')
project.export_report(project.report, 'modal_report.html')
project.save('modal.vdyn')            # opens in the app as-is
```

Every plot the app draws has a call, and `path=` (for 3-D scenes,
`screenshot=`) renders it without a window. Step 6's figure is one
line:

```python
basis = project.basis
basis.frf.plot_cmif(basis.shapes, path='cmif.png')
project.plot_mac(basis.shapes, project.other.shapes, path='crossmac.png')
project.animate(basis.shapes, mode=0, screenshot='mode1.png')
```

## Where judgment lives

Two steps ask you to look and decide:

- Which peaks are modes. The fitting screen shows the residual CMIF;
  you confirm a mode and watch what is left. In a script, `fit_modes`
  confirms the most prominent residual peak each time and stops after
  `limit` modes. The plate example sets five, because eleven response
  channels cannot reliably separate more modes in that band, and
  without a limit the loop goes on fitting peaks in the noise floor.
  The limit stands in for that judgment, so check the result against
  the CMIF. A fit made by eye can still be replayed: an interactive
  session records itself in the console as
  `fit_modes(at=[(frequency, damping), ...])`, the confirmed poles in
  the order they were taken, so the same fit runs in a script.
- Which MAC squares are matches. In the app you click them. In a
  script, `match_modes` takes each mode's best partner above a MAC
  threshold, or an explicit list of pairs (`pairs=`).

Everything else (importing, converting, computing, correlating,
reporting) runs the same code either way.

## Scaling, and when it is real

A modal fit splits residues into shape coefficients only when a drive
point was measured, that is, a response at the excitation DOF. Visual
Dynamics does this automatically when the data has one; the plate
survey measures both of its shaker points for this reason. Without a
drive point the scale is a convention: frequencies, damping, shapes
and MAC are all correct, but modal masses are not physical. The shape
set is then marked unscaled, and that is shown in the status bar, in
the mode table's modal mass column heading, and in the caption of the
report's parameter table.
