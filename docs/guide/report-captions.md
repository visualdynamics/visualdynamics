# A report plot's caption

A spectrum in a report is evidence only when a reader who was not at
the test can say what it is: which channel, at which location and in
which axis; which run, and which stretch of it; how the spectrum was
estimated; the line spacing; the window and overlap of its frames; the
degrees of freedom of its average; and the requirement drawn on it.
Those lines are the same on every plot of a kind, and every one of
them is already held by the record or the objects around it. Visual
Dynamics reads them into a caption block, so none of them is typed and
none goes stale when the analysis is redone.

The line spacing and the degrees of freedom are not formalities. A
resonance's peak height in an autospectrum depends on the spacing, and
how far a line may sit from the reference by chance depends on the
degrees of freedom ([What an average is worth](averaging-confidence.md)).
Without them, two plots cannot be compared and an excursion cannot be
judged.

## What the block says

![One control channel's ASD against its profile, the warning band and
abort limits shaded, with the caption block under it](images/report-caption.png)

The block has eight rows, always in this order:

| Row | Read from |
|---|---|
| Channel | The record's DOF and quantity, with the channel number from the channel table row of that DOF and quantity |
| Location and axis | The node and direction in the DOF, the channel table's comment, and the global axis the direction is nearest on the geometry |
| Run | The time history the spectrum was computed from, and the stretch of it averaged, in seconds on its clock |
| Estimate | ASD or CPSD, one channel or several, a linear average of how many frames, and the bands for a banded spectrum |
| Resolution | The line spacing, or the bands and the lines they were banded from |
| Window | The window, the overlap and the frame length the spectrum was averaged with |
| DOF | The equivalent degrees of freedom, discounted for overlap, beside the nominal count |
| Requirement | The specification drawn, which of its limits are drawn, and the overall rms measured beside the rms it asks for, over the band the comparison judged |

A fact nothing records is written as "not recorded" rather than left
out, so a reader sees the gap instead of assuming the row was never
needed. A spectrum imported from a file carries no averaging, so its
window, frames and degrees of freedom read "not recorded", and it
names itself as the run. A banded spectrum carries no degrees of
freedom of its own, because banding sums lines and raises the count;
its caption gives the count of each narrowband line before banding,
and says so.

The two rms values in the Requirement row are the comparison's own
(the ones the verdict table states): both over the band it judged,
with the measurement scaled as it is drawn, and the scale named in the
row when there is one.

## In a script

`Project.caption_of` reads the caption for one record, or for the
records a plot draws, from the project: the averaging from the
spectrum's provenance, the run it came from, and the channel table and
geometry linked with them.

```python
import visualdynamics

project = visualdynamics.random_vibration_run('run.nc4')
caption = project.caption_of('Time History PSDs', 0,
                             specification='Specification')
for label, said in caption.rows():
    print(f'{label}: {said}')
print(caption.text())          # the same, as one paragraph

visualdynamics.plot.plot_captioned(
    project['Time History PSDs'], 0, caption=caption,
    specification=project['Specification'], path='control_1.png')
```

`plot_captioned` draws the record against the specification as the
comparison view draws it, and puts the caption block under the plot,
a row a line with each label in bold, so a picture pasted into another
document keeps the lines that let it be checked. Without a `caption`
it reads one off the record alone, where the window and the run are
not known. `core.report_caption.record_caption` is the function under
both, for a spectrum and its averaging held outside a project.

## In a report

A plot block of a measured spectrum draws the caption block under its
figure when the block sets `'record_caption': True`, or when **Caption
from the record** is checked in the editor's pane for that block. The
pane reads the caption out beneath the check, as the page will show
it. A figure drawn a channel at a time carries each channel's block,
and the page shows the one picked. A grid of control channels says
once, under the grid, what its cells share (the run, the estimate, the
resolution, the window and the DOF), and each cell gives only its own
channel's rows. The block's own caption stays the figure's title, and
figure references still find the figure by it.

```python
report.blocks.append({'kind': 'plot', 'source': '@basis:Psd',
                      'specification': '@basis:Specification',
                      'channel': '101Z+', 'record_caption': True,
                      'caption': 'Control channel 101Z+'})
```
