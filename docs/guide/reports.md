# Reports

A report is an ordered list of blocks that refer to project objects.
It holds references, never data, so a saved report can be used as a
template for the next project.

```python
name = project.generate_report('modal')   # or 'random', 'transient',
project.export_report(name, 'modal_report.html')  # 'shock', 'sine',
                                                  # 'sysid', 'empty'
```

Each project type has its own template. *Generate Report* on the bar,
with the project row selected, builds the one that matches the type; a
Random and Sine project gets two reports, one from the random template
and one from the sine template. `'empty'` starts a blank outline.
`generate_report` also takes `marking=` and `marking_color=` (see
[Classification markings](#classification-markings)).

A project with a type also has *Automatic* on the tree bar, beside
*Link* and *Unlink*. One press computes, in the workflow's order, every
gray slot the type still expects that the loaded data can fill, and
generates the report last. For a random run, the time data and
specification give the PSDs, the octave-band PSDs and specification,
and the multiple coherence. For a modal run, the time data gives its
PSDs, FRFs and coherence, and a fitted shape set. A shock record is
filtered and its SRS, velocity and displacement are computed. A sine
sweep has its levels extracted. A system identification gets the PSDs
of both streams on shared frames and the H1 plant FRF. Anything the
data cannot supply (a geometry, photographs, a specification) stays
gray, and a computation the data cannot support (a coherence with no
drive channel) is skipped while the rest are still made. Pressed again
with nothing left to make, it adds nothing. In a script the same press
is `project.work_up()`, and it is recorded in the journal as that one
line.

The export is one self-contained HTML file: plots, animated 3D scenes,
tables and photos, each with its own interactive viewer and no
third-party code inside. It opens on a locked-down machine with no
network connection.

## Copying a figure, a table or a photo

Every figure, table and photograph has a copy button in its top-right
corner, shown when the pointer is over it. A figure is copied to the
clipboard as an image, as drawn, in the current theme and at the
screen's own resolution. A photo is copied as the image it is. A table
is copied as cells and as a table at once, so a spreadsheet pastes
cells and a document or an email pastes a table. This saves taking a
screenshot. The button works in the report editor and in the exported
file opened in a browser, and it never appears on a printed page. A
grid figure copies one cell at a time.

## Classification markings

Every report has a bold header and footer strip on each page, reading
UNCLASSIFIED by default. In the report editor the marking can be
edited. The header and footer are one value, so editing either changes
both, because a page whose top and bottom disagree about its
classification is wrong. Its color is either ink (black or white,
following the light or dark theme) or red. Clearing the text removes
the strips entirely, for work that carries no marking. The marking is
stored in the report object and in the `.vdyn` file, so it is kept
when the project is reopened and the report is exported again.

## Bindings that survive renaming

By default, blocks bind symbolically:

- `@basis:Frf` is the first FRF in the Basis group
- `@other:ShapeSet` is the first shape set outside it (the model side)
- `@any:MatchedModes` is the first one anywhere

Selectors are resolved against the project's links when the report is
rendered, so a report generated in one project works in another
whatever its objects are called, and nobody has to name anything in a
particular way. In a project with no Basis group, `@basis` falls back
to the first object of the type and `@other` to the second. Concrete
object names also work, and the editor's drop-downs offer both.

## What the modal template contains

A test summary with live values, the test geometry, the excitation and
response DOF figures, setup photos, the channel table, time and PSD
plots by quantity, the drive-point FRF magnitudes and imaginary parts
with the identified frequencies marked, the multiple coherence staged
over every channel, the CMIF with the modal model's resynthesis dashed
over it, the mode table, the auto-MAC and the animated mode shapes.
When the project has them, it also contains the test-analysis
cross-MAC, the matched-modes table, and the matched pairs animated
over each other.

Blocks whose objects the project does not have stay unbound and are
not rendered, so the outline is there to fill in.

## Live text

Markdown text blocks resolve `{{Object.field}}` references to live
values, for example sample rate, number of averages, frequency
resolution, mode count and frequency range. They also resolve
`{{figure:caption-start}}` and `{{table:...}}` cross-references, which
renumber themselves when blocks move. A reference that cannot be
resolved stays in the text as written.

```markdown
Time data was acquired on {{@basis:TimeHistory.num_channels}} channels
at {{@basis:TimeHistory.sample_rate}}, averaged over
{{@basis:TimeHistory.num_averages}} frames. See {{figure:CMIF}}.
```

## Editing a report

The editor shows the report exactly as the export will read, with the
same page and the same interactive figures, and puts every act on the
bar above it, like every other object's bar. Click a block in the page
to select it; the pane beside the page shows that block's settings. A
figure block offers its source and shapes drop-downs and its caption.
A text block opens its Markdown in an editor, and the page re-renders
as you type. With nothing selected, the pane holds the report's title
and its classification marking.

On the bar, *Insert* adds a block after the selection (Markdown, any
result the project can provide, a photo, a table). *Reference* inserts
a figure or table reference at the cursor while a text block is being
edited; the reference renumbers itself when blocks move. *Move Up*,
*Move Down* and *Delete* act on the selected block. *Export → HTML…*
writes the HTML file. None of the editor's controls appear in the
exported page, which holds only the document.

## Report features

With nothing selected, the pane also holds the **Report Features**
panel: optional additions to the report, each off until it is turned
on, chosen one by one or from a preset. With none turned on, the report
is exactly what this page describes. See
[Report features](report-features.md).

## Saving a report as a template

*Export → Report template…* on the report's bar writes the report on
its own (its blocks, their bindings, the title, the marking and the
report features, but no values) as a `.vdreport` file, by default into the application's
templates folder. A template saved there is offered by *Generate
Report* beside the built-in ones, in every project, so a house
template is one click away. Imported from anywhere else, the file adds
a Report to the project like any other import. Blocks bound
symbolically resolve against the new project's structure; a block
bound to a name the project does not have arrives as an unbound card,
ready to be pointed at another object in the pane. From a script,
`project.export(name, 'house.vdreport')` writes the template,
`project.generate_report('house')` builds a report from a template
saved in the templates folder under that name, and
`project.generate_report(path)` or `project.import_file(path)` loads
one from anywhere.
