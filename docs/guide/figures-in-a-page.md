# A figure in a page of your own

`visualdynamics.export_html` writes a figure as one HTML file that
carries its own script, stylesheet and data. That suits a file opened
on its own or framed in a slide. It does not suit a website whose
Content-Security-Policy refuses inline scripts and refuses to be
framed, which is the policy a careful site sets: there the one file
shows nothing. `visualdynamics.export_figure` writes the same figure in
two parts for such a page, the data as a JSON file and the viewers as
one script the page serves from its own origin, and the page decides
where the figure goes.

## Writing the two files

```python
import visualdynamics

visualdynamics.export_figure('spec.json', psd, specification=spec,
                             channel='101Z+', theme='light',
                             renderer='visualdynamics-figure.js')
visualdynamics.export_figure('modes.json', geometry=geometry,
                             shapes=modes, theme='light')
```

The data file holds one figure, the payload the HTML file would have
carried inline, and a format number. The renderer is the same script
for every figure: the viewers of the HTML file, made to draw into an
element instead of a whole page. It is written only when `renderer` is
given, and a site needs one copy of it. It carries the package's MIT
license at its head, so the copy is complete wherever it is served.

## Mounting it

The page loads the renderer as an ordinary script from its own site and
hands it an element:

```html
<div id="spec"></div>
<script src="/js/visualdynamics-figure.js"></script>
<script src="/js/figures.js"></script>
```

```javascript
// figures.js
const element = document.getElementById('spec');
VisualDynamicsFigure.load(element, '/data/spec.json', {pageScroll: true})
  .catch(() => { /* keep whatever the element held, a picture say */ });
```

`load(element, url, options)` fetches the data and draws it, and returns
a promise; `mount(element, data, options)` draws data the page already
has. Either replaces the element's contents, so a still image placed
inside it shows until the figure is drawn and stays if anything fails.
Both refuse data written for a different format than the renderer's
(`VisualDynamicsFigure.format`), rather than draw it wrongly: when the
package changes the payload, the renderer and the data are written
again together.

The figure keeps to its element. Its styles are scoped under the class
the renderer gives the element (`vdfig`), so it styles nothing of the
page around it, and it leaves out what a page of its own would show:
the title, the theme toggle and the figure number, since the host page
numbers its figures. If the page's policy gives scripts a nonce, give
the renderer's script tag one; the stylesheet it adds carries the same
nonce.

`{pageScroll: true}` is for a figure set among text. The wheel then
scrolls the page past the figure, as a reader expects, and zooms only
with Ctrl or Cmd held, which is also how a trackpad's pinch arrives;
on a touch screen a vertical swipe scrolls the page and a sideways one
pans the plot. The hint under the figure says so. Without it the wheel
zooms, as in a page of its own.

With `fill=True` the figure fills its element, which then needs a
height of its own; by default it takes the height a report page gives
it.

## Making the data smaller

A figure carries its data at full resolution, so a long record makes a
large file. Two arguments, taken by `export_html` as well, trim it:

```python
visualdynamics.export_figure('run.json', history, channels=['101Z+', '102Z+'],
                             max_points=4000)
```

`channels` keeps only the records at those response DOFs, and the
specification's records likewise; a channel the data does not hold is
refused, naming the ones it does. `max_points` thins every curve of a
plot to at most that many points by keeping, slice by slice, the
smallest and largest value of every curve, so no peak or dip is lost
and the first and last points stay where they were. Curves drawn on one
abscissa, a measurement and the limits shaded about it, are thinned to
the same points, so a band still lies between points that are above one
another. A density drawn stepped across its bins is drawn as a line
once thinned, since the bin edges no longer describe the bins kept.
The figure's note says that it was thinned and from how many points,
as every thinned figure's does. The readouts beside a comparison, the
lines out and the RMS error, are worked out from the whole record
before it is thinned. A scene is not thinned by `max_points`: choose its
modes by passing fewer shapes.
