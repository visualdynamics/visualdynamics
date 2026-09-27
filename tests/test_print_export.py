"""Plots written for print keep their proportions.

pyqtgraph's ImageExporter, asked for more pixels than a plot's logical
width, scaled its items unevenly: at three times the width the legend's
labels came out a third of their set size and QFont tick labels half
again too large (the band-average paper's print figures, 2026-09-26).
`plot.save_image` paints the laid-out plot at a device ratio instead, so
everything scales together, and says its resolution in the file.
"""

from __future__ import annotations

import numpy as np
from test_exceedance_shading import measured, spec

_ALIVE = []


def _plot(qt_app, size=(420, 300)):
    import pyqtgraph as pg

    from visualdynamics.plot import build_plots

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    layout.resize(*size)
    build_plots(layout, [('Specification', spec(abort_upper=4.0,
                                                 abort_lower=0.25), None),
                         ('Response', measured(np.linspace(20, 2000, 120)),
                          None)], theme='light')
    layout.show()
    for _ in range(10):
        qt_app.processEvents()
    plot = next(item for item in layout.ci.items
                if hasattr(item, 'listDataItems'))
    return layout, plot


def _ink(image, rect, ratio):
    """The share of a region's pixels that are not the background."""
    from PySide6.QtGui import QColor

    background = QColor('#ffffff')
    x0, y0 = int(rect.left() * ratio), int(rect.top() * ratio)
    x1, y1 = int(rect.right() * ratio), int(rect.bottom() * ratio)
    inked = total = 0
    step = max(1, int(ratio))
    for x in range(x0, x1, step):
        for y in range(y0, y1, step):
            c = image.pixelColor(x, y)
            total += 1
            if abs(c.lightness() - background.lightness()) > 40:
                inked += 1
    return inked / max(total, 1)


def test_the_legend_prints_at_its_set_size(qt_app, tmp_path):
    """At 288 dpi (three device pixels to a logical one) the legend's
    text covers the same share of its box as it does on screen — the
    exporter's third-size labels covered about a ninth of it."""
    from PySide6.QtGui import QImage

    from visualdynamics.plot import render_image, save_image
    from visualdynamics.theme import theme as resolve_theme

    layout, plot = _plot(qt_app)
    box = layout.mapFromScene(plot.legend.sceneBoundingRect()).boundingRect()
    screen = render_image(layout, 1.0, resolve_theme('light')['plot_background'])
    path = save_image(layout, tmp_path / 'print.png', dpi=288, theme='light')
    printed = QImage(path)
    assert (printed.width(), printed.height()) == (3 * layout.width(),
                                                    3 * layout.height())
    on_screen, in_print = _ink(screen, box, 1.0), _ink(printed, box, 3.0)
    assert on_screen > 0.02, 'the legend has text to measure'
    assert 0.6 < in_print / on_screen < 1.6, (on_screen, in_print)


def test_a_print_image_says_its_resolution(qt_app, tmp_path):
    from PySide6.QtGui import QImage

    from visualdynamics.plot import save_image

    layout, _plot_item = _plot(qt_app, size=(328, 200))     # 3.42 in wide
    image = QImage(save_image(layout, tmp_path / 'p.png', dpi=300))
    assert image.width() == round(328 * 300 / 96)
    assert abs(image.dotsPerMeterX() * 0.0254 - 300) < 1
    assert abs(image.width() / (image.dotsPerMeterX() * 0.0254) - 3.42) < 0.01


def test_a_standalone_plot_writes_at_the_export_setting(qt_app, tmp_path):
    """`plot.EXPORT_DPI` is how a script asks every standalone .png for
    print; unset, a plot is one pixel per logical pixel, as it was."""
    from PySide6.QtGui import QImage

    import visualdynamics.plot as plotting

    series = [('S', spec(abort_upper=4.0), None)]
    plain = QImage(plotting.plot_series(series, path=tmp_path / 'a.png',
                                        size=(300, 200), show=False))
    assert (plain.width(), plain.height()) == (300, 200)
    before = plotting.EXPORT_DPI
    plotting.EXPORT_DPI = 192
    try:
        sharp = QImage(plotting.plot_series(series, path=tmp_path / 'b.png',
                                            size=(300, 200), show=False))
    finally:
        plotting.EXPORT_DPI = before
    assert (sharp.width(), sharp.height()) == (600, 400)


# ---- the bar charts: a key, and a count that fits (2026-09-26) ----------


def _chart(qt_app, summary='over', low=-3.0, high=3.0):
    import pyqtgraph as pg

    from visualdynamics.plot import legend_below
    from visualdynamics.plot.bars import BarChart
    from visualdynamics.theme import theme as resolve_theme

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    plot = layout.addPlot(row=0, col=0)
    colors = resolve_theme('light')
    chart = BarChart(plot, [('101Z+', 4.5), ('102Z+', -1.0), ('103Z+', -4.0)],
                     colors, low=low, high=high, label='RMS error', units='dB',
                     summary=summary)
    legend = legend_below(layout, plot, 0, colors)
    chart.key(legend)
    return chart, plot, legend


def test_a_bar_chart_keys_its_colors(qt_app):
    """No key said what the gray, red and blue bars or the shaded ground
    meant; `key` names them, each swatch the brush it names."""
    chart, _plot, legend = _chart(qt_app)
    names = [label.text for _sample, label in legend.items]
    assert names == ['within tolerance', 'over tolerance', 'under tolerance',
                     'past +3 dB', 'past −3 dB']
    brushes = {label.text: sample.item.opts['brush'].color().name()
               for sample, label in legend.items}
    assert brushes['over tolerance'] == chart._brush(4.5).color().name()
    assert brushes['under tolerance'] == chart._brush(-4.0).color().name()
    assert brushes['within tolerance'] == chart._brush(0.0).color().name()


def test_a_one_sided_chart_keys_one_side(qt_app):
    _unused, _plot, legend = _chart(qt_app, low=10.0, high=None)
    names = [label.text for _sample, label in legend.items]
    assert names == ['within tolerance', 'over tolerance', 'past +10 dB']


def test_the_count_can_go_in_the_title(qt_app):
    """At a quarter of a page the count over the bars ran into them; in
    the title it cannot."""
    chart, plot, _legend = _chart(qt_app, summary='title')
    assert chart.summary is None
    assert '2 of 3 channels outside -3 to 3dB' in plot.titleLabel.text
    chart, plot, _legend = _chart(qt_app, summary='over')
    assert chart.summary is not None and not plot.titleLabel.text
    assert '2 of 3 channels' in chart.summary.textItem.toPlainText()


# ---- 3-D scenes at print resolution (2026-09-26) -------------------------


def test_a_print_plotter_is_the_printed_size_at_its_dpi():
    """The window is the figure's inches times its dpi, and the render
    window's DPI makes a text actor's points print as points: 72 is VTK's
    one-to-one, so 288 dpi against the screen's 96 is 216."""
    from visualdynamics.viz.geometry import print_plotter

    plotter, scale = print_plotter((3.0, 2.25), 288)
    try:
        assert tuple(plotter.window_size) == (864, 648)
        assert plotter.ren_win.GetDPI() == 216
        assert scale == 3.0
    finally:
        plotter.close()


def _scene(dpi):
    from visualdynamics.demo import barc
    from visualdynamics.viz.geometry import plot_geometry

    image = plot_geometry(barc.upright(barc.geometry(0.25)),
                          screenshot=str(_scene.dir / f'g{dpi}.png'),
                          theme='light', size_in=(3.0, 2.25), dpi=dpi)
    return np.asarray(image, dtype=float).mean(axis=2)


def test_a_print_render_is_the_screen_render_enlarged(tmp_path):
    """Rendered at three times the resolution and shrunk back, the scene
    is the screen-size one: nodes, lines and the bounds' labels (sized
    in screen pixels, by the cube axes' own screen size) all scaled.
    Unscaled, they were a third of their size in print."""
    _scene.dir = tmp_path
    screen, printed = _scene(96), _scene(288)
    h, w = (printed.shape[0] // 3) * 3, (printed.shape[1] // 3) * 3
    shrunk = printed[:h, :w].reshape(h // 3, 3, w // 3, 3).mean(axis=(1, 3))
    h, w = min(shrunk.shape[0], screen.shape[0]), min(shrunk.shape[1], screen.shape[1])
    difference = np.abs(shrunk[:h, :w] - screen[:h, :w]).mean()
    assert difference < 14.0, difference


def _red_height(image):
    """The tallest red glyph in an image, in pixels."""
    from scipy import ndimage

    image = np.asarray(image).astype(int)
    red = (image[..., 0] > 150) & (image[..., 1] < 100) & (image[..., 2] < 100)
    labeled, _count = ndimage.label(red)
    heights = [piece[0].stop - piece[0].start
               for piece in ndimage.find_objects(labeled)
               if 3 < piece[0].stop - piece[0].start < 300
               and 2 < piece[1].stop - piece[1].start < 300]
    return max(heights, default=0)


def test_the_bounds_labels_print_as_the_other_labels_do():
    """The box's labels are sized by the cube axes' screen size — the
    font size and the render window's DPI both left them at screen size,
    about 2 pt in print — and set by the plain scale they came out at 14
    to 16 pt, twice the scene's other labels (the band-average paper,
    2026-09-27). At 300 dpi a title now stands within a quarter of a
    13-point label's height."""
    import pyvista as pv

    from visualdynamics.theme import theme as resolve_theme
    from visualdynamics.viz.geometry import annotate_scene, print_plotter

    plotter, scale = print_plotter((4.5, 3.25), 300)
    plotter.set_background('white')
    plotter.add_mesh(pv.Cube(x_length=12, y_length=12, z_length=4),
                     opacity=0.0)
    colors = dict(resolve_theme('light'), scene_text='red')
    annotate_scene(plotter, '', colors, orientation=False, scale=scale)
    box = plotter.renderer.cube_axes_actor
    for axis in ('X', 'Y', 'Z'):
        getattr(box, f'Set{axis}Title')('XXXX')
    box.SetFlyModeToOuterEdges()
    title = _red_height(plotter.screenshot(return_img=True))
    plotter.close()
    label, _scale = print_plotter((4.5, 3.25), 300)
    label.set_background('white')
    label.add_point_labels(np.zeros((1, 3)), ['XXXX'], font_size=13,
                           text_color='red', shape=None, show_points=False,
                           always_visible=True)
    point = _red_height(label.screenshot(return_img=True))
    label.close()
    assert point > 10
    assert 0.75 < title / point < 1.33, (title, point)


def test_a_print_figure_can_leave_the_box_out(tmp_path):
    """`bounds` and `orientation` pass through the screenshot calls, so a
    print figure can drop the box (the band-average paper, 2026-09-27)."""
    from visualdynamics import mesh
    from visualdynamics.viz import geometry as scene

    plate = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5)
    seen = []
    real = scene.annotate_scene

    def spy(plotter, *args, **kwargs):
        seen.append((kwargs.get('bounds'), kwargs.get('orientation')))
        return real(plotter, *args, **kwargs)

    scene.annotate_scene = spy
    try:
        scene.plot_geometry(plate, screenshot=str(tmp_path / 'g.png'),
                            size_in=(2.0, 1.5), dpi=150, bounds=False,
                            orientation=False)
    finally:
        scene.annotate_scene = real
    assert seen == [(False, False)]


def test_a_thin_axis_keeps_its_title_and_drops_its_numbers():
    """A plate's thickness is an axis a few pixels long: its numbers
    stacked into a smudge, five or two (the band-average paper,
    2026-09-27). Every axis a tenth of the model or longer keeps them."""
    from visualdynamics.viz.geometry import _label_counts

    assert _label_counts((0, 12, 0, 12, -0.5, 0)) == {
        'show_xlabels': True, 'show_ylabels': True, 'show_zlabels': False}
    assert all(_label_counts((0, 12, 0, 6, 0, 3)).values())
