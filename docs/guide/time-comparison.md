# Comparing two sets of time responses

A model's predicted time responses are compared with a test's measured
ones station by station: the same DOF, the same quantity, one trace
from each set. The comparison puts each pair on one clock, draws the
two traces over each other, and scores how well their shapes agree
with the time response assurance criterion (TRAC).

## The scores

For a pair of traces, the measured one \(a\) and the predicted one \(b\),
sampled on one clock inside the scoring window,

\[
\mathrm{TRAC} = \frac{(a \cdot b)^2}{(a \cdot a)(b \cdot b)}
\]

TRAC runs from 0 to 1 and reaches 1 when the two traces have the same
shape. Because the inner product is squared and normalized, a trace
with its sign reversed, or scaled by any constant, scores the same as
the original. Three more numbers are given beside it so that those
cases show:

- the signed correlation, \((a \cdot b) / \sqrt{(a \cdot a)(b \cdot b)}\),
  which is negative for a sign reversal;
- the RMS ratio, \(\|b\| / \|a\|\), which is the amplitude difference;
- sTRAC, \((a \cdot b) / (b \cdot b)\), the least-squares factor that
  multiplies the second trace to fit the first. Its ideal value is 1:
  a value of \(-2\) means the second trace is half the size of the
  first and the wrong way round.

A trace that is zero, constant or not finite inside the window has no
score, and its row in the table says why instead.

## Pairing and the shared clock

Select the two sets of time histories in the tree (two histories, the
records of one, or two of the tree's history groups) and choose
**TRAC / Overlay** on the data bar. Each response record is paired with
the one record in the other set that has the same signed DOF and the
same quantity. Force and other reference channels are left out. A DOF
that has no partner, or more than two records, is refused with the DOF
named, rather than paired by guessing. Two sets whose node numbers
differ are paired by hand with **Edit pairs…**, a table with one row a
pair, which copies, pastes and fills like every other table. Records
in different quantities, or with units not yet declared, cannot be
paired.

The pairs are put on the coarsest of the sources' clocks, inside the
time the sources have in common; the other traces are interpolated
linearly onto it. Nothing else is done to them: no lag is removed, no
sign or gain is fitted, and no filter is applied. A source sampled
much faster than the shared clock is interpolated without an
anti-alias filter, so filter it first (**Filter**, below) if it holds
content above the coarser clock's Nyquist frequency.

## The comparison pane

The pane beside the plot has the scoring window, **TRAC start** and
**TRAC end**, in seconds, and the score table, one row a station with
its TRAC and sTRAC. The plot draws the station picked in the table,
both traces over each other. With the data bar's 3-D form on, every
station is drawn at once, the two sets side by side, each station
labeled with its TRAC.

**Save comparison** adds a *Time Response Comparison* to the project:
the paired traces on their shared clock, with the pairs, the sources
and the window kept with it. Selecting it again shows the same pane,
where **Save TRAC window** keeps a new window. The comparison is
derived from its sources, so it is marked for refresh when a source
changes, and it follows a source that is renamed.

## Scores on the geometry

Link the saved comparison to the test's geometry and choose a
**Geometry correlation**: the TRAC or the sTRAC of each station as a
colored point at its node, or as a contour over a finite element
geometry. The 3-D view comes up beside the plot with the scores drawn
on it and their average in the corner. Click a station to select its
row in the score table and plot its two traces; clicking a node that
has more than one scored DOF steps through them. **FEM overlay** picks the
geometry the contour is drawn on, drawn translucent under the
stations; each of its nodes takes the score of the nearest station, or
an inverse-distance-squared blend of all the stations with the
*smooth* contour.

## Batch filtering and rescaling

**Filter** and **Rescale** also apply to several time histories at
once. Select them all, and the reading's pane shows the first one;
**Apply** makes a filtered or rescaled copy of every one with the same
settings, each checked before any is added. The sources are left as
they were. With View → Group Time Histories on, the tree gathers the
histories under one row per imported file, and the filtered and
rescaled copies under rows of their own. Double-click a row to rename
it; the name is saved with the project.

## Paired time traces from a Universal file

Some exporters write each measurement as two dataset 58 records, a
*Response Time Trace* and a *Reference Time Trace*. Each such pair is
imported as one time history with its response and reference roles
set, named for the two DOFs (`Time History - 730Z- - ref 1013Z+`).
Exporters that write a channel number in the reference field instead
of the reference's node are corrected at import: the question that
comes up shows the reference DOF the file names and takes the right
node. The direction is kept.

## From a script

    import visualdynamics

    project = visualdynamics.Project.open('test.vdyn')
    filtered = project.filter_histories('Test', 'Model')
    name = project.compare_time_histories(*filtered, start=0.1, stop=0.6)
    project.set_time_comparison_window(name, 0.2, 0.6)

    from visualdynamics.core.time_compare import scores

    for row in scores(project[name]):
        print(row['dof'], row['trac'], row['strac'])

`compare_time_histories` takes `pairs=` as a list of
`[first name, record, second name, record]` for hand-made pairs, and
`records=` to limit the automatic pairing to some records of each
source. `scores` reads a saved comparison over its own window, or over
`start` and `stop` when they are given. `unv.load(path,
reference_node=1013)` and `project.import_file(path,
reference_node=1013)` set the reference node of a Universal file's
paired traces.
