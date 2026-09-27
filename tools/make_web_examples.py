"""Build the website's interactive examples from the plate project.

    ./.venv/bin/python tools/make_web_examples.py

Three figures, written to `web/launch/examples/` as self-contained
pages the site embeds in iframes, all from one modal survey of a
plate and the model of it:

- the measured time data, on a 3-D stage
- the measured plant, as frequency response functions on the same stage
- the finite element modes, animated and rotatable

They are produced by the **report renderer**, not by a second drawing
path built for the web. That is the point: what a visitor rotates on
the website is the same figure, from the same payload, that a report
exported from the application would carry. A separate web renderer
would be a second implementation of the same rule, free to drift
(PRINCIPLES.md, 9).

**Deliberately smaller than a report.** A report must not reduce the
data it reports — a figure that has been thinned cannot answer a
question asked of it later — but a web page is an advertisement for
the tool rather than evidence about a test, and 140 MB of it would
be an advertisement nobody waits for. So a handful of channels are
selected and `MAX_FIGURE_POINTS` is lowered, which makes the figures
say so themselves: the thinning note is part of the payload and rides
into the page.

The source is `stressdata/`, which is not in the repository — the
generated pages are committed instead, so the site can be rebuilt
without it and only someone regenerating the examples needs the data.
"""

from __future__ import annotations

import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'web' / 'launch' / 'examples'
SOURCE = ROOT / 'stressdata' / 'plate_projects' / 'modal.vdyn'

#: points one web figure may carry across all its curves. Two orders
#: below the report's own budget: the report is evidence, this is a
#: shop window, and the figures label themselves as thinned.
#:
#: Set by how the figure *handles*, not by how it looks still. The
#: stage redraws every curve on every pointer move, so the budget is
#: what one drag frame can afford — measured at about 9 ms a draw for
#: 65,000 points on a retina display, which is fine for one frame and
#: visibly slow when a trackpad sends five moves between two of them
#: (Brandon, 2026-08-25). A third of that is smooth, and no detail is
#: lost that a 900-pixel figure could have shown: 2,000 points to a
#: curve is already more than twice the pixels it is drawn across, and
#: the thinner keeps every extreme, so peaks survive it.
WEB_FIGURE_POINTS = 20_000

#: how many of each to show. Enough to look like real test data, few
#: enough that the page arrives and turns smoothly.
FRF_CURVES = 10
TIME_CHANNELS = 6
MODES = 6



#: Appended to each generated page. The pages stay complete documents —
#: open one on its own and it is an ordinary report figure — but when
#: the website frames one, the report's page furniture (margins, the
#: empty title rule, the theme toggle) is the host page's job, and the
#: figure has to be able to say how tall it is so the frame can be cut
#: to fit. Without that the visitor gets a scrollbar around a picture,
#: which is the one thing a shop window must not have.
#:
#: It lives here rather than in the renderer because it is about being
#: embedded in *this website*, which the renderer has no business
#: knowing. The dark theme is forced for the same reason: the site has
#: one palette, and a visitor whose OS prefers light would otherwise
#: get dark-on-dark ink through the transparent background.
EMBED = """
<style>
/* `color-scheme: dark` makes the browser paint its own canvas black,
   so the root needs saying transparent explicitly or the figure sits
   on a black slab instead of on the site's own background. */
html.embedded { background: transparent; }
body.embedded { padding: 0; max-width: none; background: transparent;
                display: flow-root; }
body.embedded h1, body.embedded .themetoggle { display: none; }
body.embedded section:first-of-type { margin-top: 0; }
body.embedded section:last-of-type { margin-bottom: 0; }
</style>
<script>
if (window.parent !== window) {
  document.documentElement.dataset.theme = 'dark';
  document.documentElement.classList.add('embedded');
  document.body.classList.add('embedded');
  /* the body's own box, not the document's: once the frame has been
     cut to fit, scrollHeight is the viewport height and would never
     report a shrink. */
  const tell = () => parent.postMessage(
    { visualdynamics: 'height',
      height: Math.ceil(document.body.getBoundingClientRect().height) }, '*');
  addEventListener('load', tell);
  new ResizeObserver(tell).observe(document.body);
}
</script>
</body>
"""


def trimmed(data, keep):
    """A copy of `data` holding only the records in `keep`."""
    fields = {'response_dof': [data.response_dof[i] for i in keep],
              'ordinate_dim': [data.ordinate_dim[i] for i in keep],
              'ordinate_unit': [data.ordinate_unit[i] for i in keep]}
    if data.reference_dof is not None:
        fields['reference_dof'] = [data.reference_dof[i] for i in keep]
        fields['reference_unit'] = [data.reference_unit[i] for i in keep]
    return type(data)(data.abscissa, data.ordinate[keep], **fields)


def spread(count, wanted):
    """`wanted` indices spread across `count`, so the sample is of the
    whole set rather than of whichever channels came first."""
    step = max(1, count // wanted)
    return list(range(0, count, step))[:wanted]


def of_one_quantity(data):
    """The rows measuring whatever most of them measure.

    A survey's time history holds the hammer's force alongside the
    responses, and a stage draws one quantity at a time — so a sample
    taken across all the rows arrives with a record or two the figure
    then has to say it is not showing. Choosing from the accelerations
    alone is the same sample without the apology."""
    dims = list(data.ordinate_dim)
    wanted = max(set(dims), key=dims.count)
    return [i for i, dim in enumerate(dims) if dim == wanted]


def build():
    import visualdynamics
    import visualdynamics.report as report_module
    from visualdynamics.core.report import Report

    if not SOURCE.exists():
        raise SystemExit(
            f'{SOURCE} is absent. The examples are generated from the '
            'plate project, which lives in stressdata/ and is not in '
            'the repository; the pages they produce are committed, so '
            'this only needs running when the examples change.')

    source = visualdynamics.Project.open(str(SOURCE))
    demo = visualdynamics.Project('Examples')

    frfs = source['FRF']
    demo.add('FRFs', trimmed(frfs, spread(frfs.num_records, FRF_CURVES)))
    history = source['Time History']
    responses = of_one_quantity(history)
    demo.add('Time', trimmed(history, [responses[i] for i in
                                       spread(len(responses), TIME_CHANNELS)]))
    demo.add('Geometry', source['FEM Geometry'])

    # the elastic modes: a free-free solution opens with six rigid-body
    # modes at 0 Hz, which animate as the whole aircraft translating and
    # say nothing about the structure
    modes = copy.deepcopy(source['FEM Shape Set'])
    elastic = [i for i, f in enumerate(modes.frequency) if f > 1.0][:MODES]
    modes.delete_modes([i for i in range(modes.num_shapes)
                        if i not in elastic])
    demo.add('Modes', modes)
    demo.link('Geometry', 'Modes')

    pages = {
        'time.html': {
            'kind': 'plot', 'mode': 'stage', 'source': 'Time',
            'caption': 'Measured time data, one channel per station'},
        'frfs.html': {
            'kind': 'plot', 'mode': 'stage', 'source': 'FRFs',
            'caption': 'Measured frequency response functions, one '
                       'channel per station'},
        'modes.html': {
            'kind': 'scene', 'geometry': 'Geometry', 'shapes': 'Modes',
            'caption': 'Finite element modes of the plate — pick one '
                       'from the list, and drag to turn it'},
    }

    held = report_module.MAX_FIGURE_POINTS
    report_module.MAX_FIGURE_POINTS = WEB_FIGURE_POINTS
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        objects = dict(demo.items())
        for filename, block in pages.items():
            report = Report('', [block])
            report.marking = ''      # no classification banner on a shop window
            html = report_module.render_html(
                report, objects, visualdynamics.SI, links=demo.links)
            assert html.count('</body>') == 1
            html = html.replace('</body>', EMBED.strip() + '\n')
            (OUT / filename).write_text(html, encoding='utf-8')
            size = len(html.encode('utf-8')) // 1024
            print(f'  {filename:14} {size:5d} KB')
    finally:
        report_module.MAX_FIGURE_POINTS = held
    return 0


#: the BARC figure's element size, inches: the website draws the model
#: a quarter inch coarse where the comparison is made at an eighth — its
#: modes differ by about a percent, and the page by a factor of four
BARC_WEB_SIZE = 0.25
BARC_MODES = 8


def build_barc(size: float = BARC_WEB_SIZE, write: bool = True) -> int:
    """The BARC's finite element modes, animated: the model built from
    planes (`visualdynamics.demo.barc`), its first elastic modes. Needs
    nothing outside the repository."""
    import visualdynamics
    import visualdynamics.report as report_module
    from visualdynamics.core.report import Report
    from visualdynamics.demo import barc

    model = barc.build(size)
    shapes = model.eigensolution(maximum_frequency=1300.0)
    elastic = [i for i, f in enumerate(shapes.frequency) if f > 1.0][:BARC_MODES]
    shapes.delete_modes([i for i in range(shapes.num_shapes)
                         if i not in elastic])
    geometry, shapes = barc.upright(barc.geometry(size), shapes)
    demo = visualdynamics.Project('Examples')
    demo.add('BARC', geometry)
    demo.add('Modes', shapes)
    demo.link('BARC', 'Modes')
    report = Report('', [{
        'kind': 'scene', 'geometry': 'BARC', 'shapes': 'Modes',
        'caption': 'Finite element modes of the BARC, built from planes — '
                   'pick one from the list, and drag to turn it'}])
    report.marking = ''
    html = report_module.render_html(report, dict(demo.items()),
                                     visualdynamics.SI, links=demo.links)
    html = html.replace('</body>', EMBED.strip() + '\n')
    size_kb = len(html.encode('utf-8')) // 1024
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'barc.html').write_text(html, encoding='utf-8')
    print(f'  barc.html      {size_kb:5d} KB  ({model.num_nodes} nodes)')
    return size_kb


if __name__ == '__main__':
    build_barc()
    sys.exit(build())
