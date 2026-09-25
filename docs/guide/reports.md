# Reports

A report is an ordered list of blocks that reference project objects.
It holds references, never data, which is what makes a saved report a
template for the next project.

```python
name = project.generate_report('modal')   # or 'random', 'transient',
project.export_report(name, 'modal_report.html')  # 'shock', 'sine', 'mixed',
                                                  # 'sysid', 'empty'
```

Each project type has its template — *Generate Report* on the bar,
with the project row selected, builds the one that matches — and
`'empty'` starts a blank outline.

The export is one self-contained HTML file: plots, animated 3D
scenes, tables, photos, all with their own interactive viewers and
nothing third-party inside. It opens on a locked-down machine with no
network.

## Copying a figure

Every figure has a copy button in its top-right corner, shown when the
pointer is over it. It puts the figure on the clipboard as an image,
as drawn — in the current theme, at the screen's own resolution — for
pasting into a document, a chat or an email without a screenshot. It
works in the report editor and in the exported file open in a browser
alike, and never appears on a printed page. A grid figure copies a cell
at a time.

## Classification markings

Every report carries a bold header and footer strip on each page,
**UNCLASSIFIED** by default. In the report editor the marking is
editable — the header and footer are one value, so editing either
changes both, because a page whose top and bottom disagree about its
classification is wrong twice — and its color is either ink (black
or white, following the light or dark theme) or red. Clearing the
text removes the strips entirely, for work that carries no marking.
The marking rides in the report object and the `.vdyn` file, so it
survives export, reopening and re-export.

## Bindings that survive renaming

Blocks bind **symbolically** by default:

- `@basis:Frf` — the first FRF in the Basis group
- `@other:ShapeSet` — the first shape set outside it (the model side)
- `@any:MatchedModes` — the first anywhere

Selectors resolve against the project's links at render time, so a
report generated in one project works in another whatever the objects
are called, and nobody has to name anything in particular. Concrete
names still work; the editor's drop-downs offer both.

## What the modal template contains

Test summary with live values, the test geometry, excitation and
response DOF figures, setup photos, the channel table, time and PSD
plots by quantity, drive-point FRF magnitude and imaginary parts with
the identified frequencies marked, the multiple coherence staged over
every channel, the CMIF with the modal
model's resynthesis dashed over it, the mode table, the auto-MAC, the
animated mode shapes, and — when the project has them — the
test-analysis cross-MAC, the matched-modes table, and the matched
pairs animated over each other.

Blocks whose objects the project lacks stay unbound and simply do not
render, so the outline is there to fill in.

## Live text

Markdown text blocks resolve `{{Object.field}}` references — sample
rate, number of averages, frequency resolution, mode count, band —
and `{{figure:caption-start}}` / `{{table:...}}` cross-references
that renumber themselves when blocks move.

```markdown
Time data was acquired on {{@basis:TimeHistory.num_channels}} channels
at {{@basis:TimeHistory.sample_rate}}, averaged over
{{@basis:TimeHistory.num_averages}} frames. See {{figure:CMIF}}.
```

## Editing a report

The editor shows the report exactly as the export will read — the same
page, the same interactive figures — and puts every act on the bar
above it, as every other object's bar does. Click a block in the page
to select it; the pane beside the page shows what that block has to
say. A figure block offers its source and shapes drop-downs and its
caption; a text block opens its Markdown in an editor, and the page
re-renders as you type; with nothing selected the pane holds the
report's title and its classification marking. On the bar: *Insert*
adds a block after the selection (Markdown, any result the project can
give, a photo, a table); *Reference* inserts a figure or table
reference at the cursor while a text block is being edited — the token
renumbers itself when blocks move; *Move Up*, *Move Down* and *Delete*
act on the selected block; *Export* writes the HTML file. Nothing typed
here reaches the exported page as chrome: the export is the document
alone.

## Saving a report as a template

*Export → Report template…* on the report's bar writes the report on
its own — its blocks, their bindings, the title and the marking, no
values — as a `.vdreport` file, by default into the application's
templates folder. A template saved there is offered by Generate Report
beside the built-in ones, in every project, so a house template is one
click away; imported from anywhere else, the file adds a Report to the
project like any other import. Blocks bound symbolically resolve
against the new project's structure; one bound to a name the project
lacks arrives as an unbound card, ready to be repointed in the pane.
The same from a script: `project.export(name, 'house.vdreport')` and
`project.generate_report('house')` or `project.import_file(path)`.
