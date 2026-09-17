"""The console tab: the session as Python, at the bottom edge.

Piece three (Brandon, 2026-08-30): a tab, not a menu — collapsed by
default, one click to expand this sitting's journal, one to put it
away. Read-only here; typing into it is a later piece.
"""

from __future__ import annotations

import numpy as np

import visualdynamics


def _record(window, pump):
    t = np.arange(4096) / 1024.0
    history = visualdynamics.TimeHistory(
        t, np.random.default_rng(5).standard_normal((2, len(t))),
        response_dof=['101Z+', '104Z+'], ordinate_dim='acceleration')
    window.add_object('Run', history)
    item = window._item_for_object('Run')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    return history


def test_the_console_starts_collapsed_behind_its_tab(window):
    assert not window.console.showing
    assert not window.console.view.isVisibleTo(window.console)
    assert 'Console' in window.console.tab.text()


def _flank_ink(tab, image, side, wanted):
    """The pixels of one chevron's flank that `wanted` accepts."""
    flank = tab.FOOT + tab.INSET + tab.GLYPH + tab.GAP // 2
    columns = (range(flank) if side == 'left'
               else range(image.width() - flank, image.width()))
    return {(x, y) for y in range(image.height()) for x in columns
            if wanted(image.pixelColor(x, y))}


def test_the_tab_wears_a_chevron_each_side_of_the_word(window, pump):
    """To the design (Brandon, 2026-09-03): a flared tab, the word in
    the middle, a chevron either side pointing where a click will move
    the edge — up while closed, down once open — in the palette's text
    color so a theme change recolors them."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor, QPalette

    tab = window.console.tab
    assert tab.text() == 'Console'
    tab.resize(tab.sizeHint())
    pump()

    # the shape: shoulders cut away at the top corners, feet filled in
    # at the bottom ones — a rectangle would hold all four
    outline = tab.outline()
    w, h = tab.width(), tab.height()
    assert not outline.contains(QPointF(1, 1))
    assert not outline.contains(QPointF(w - 1, 1))
    # the feet: material in the corner outside the fillet's circle,
    # so just inside the side wall at the base is tab
    assert outline.contains(QPointF(tab.FOOT - 1, h - 1))
    assert outline.contains(QPointF(w - tab.FOOT + 1, h - 1))
    assert not outline.contains(QPointF(1, h - 1)), 'the fillet'
    assert outline.contains(QPointF(w / 2, h / 2))

    def dark(color):
        # ink is opaque and dark: the tab is translucent outside its
        # outline (2026-09-04), and a clear pixel reads as black
        return color.alpha() > 128 and color.lightness() < 128

    def points(ink):
        # a chevron's tip is where its ink gathers on the centreline;
        # the rows carry the same ink at both ends
        ys = [y for _, y in ink]
        cx = sum(x for x, _ in ink) / len(ink)
        quarter = (max(ys) - min(ys)) / 4

        def spread(rows):
            xs = [abs(x - cx) for x, y in ink if rows(y)]
            return sum(xs) / len(xs)

        top = spread(lambda y: y < min(ys) + quarter)
        bottom = spread(lambda y: y > max(ys) - quarter)
        return 'up' if top < bottom else 'down'

    image = tab.grab().toImage()
    left, right = (_flank_ink(tab, image, side, dark)
                   for side in ('left', 'right'))
    assert left and right, 'a chevron each side'
    mirrored = {(w - 1 - x, y) for x, y in right}
    assert len(left & mirrored) > 0.8 * len(left), 'the same chevron'
    assert points(left) == 'up', 'closed: away from the edge'
    tab.setChecked(True)
    pump()
    image = tab.grab().toImage()
    assert points(_flank_ink(tab, image, 'left', dark)) == 'down', 'open'

    palette = tab.palette()
    palette.setColor(QPalette.ColorRole.ButtonText, QColor('#c02020'))
    tab.setPalette(palette)
    pump()
    image = tab.grab().toImage()
    red = _flank_ink(tab, image, 'left',
                     lambda c: c.red() > c.green() + 60)
    assert red, 'recolored with the palette'


def test_the_tab_is_a_tab_and_not_a_bar(window, pump):
    """Centered, its own width, and no reserved strip: the first cut
    kept a full-width layout row for the tab and the empty row read
    as a gray bar across the window (Brandon, 2026-08-31). The tab
    floats over the views' bottom edge instead."""
    window.show()
    window.resize(900, 700)
    pump()
    host = window.centralWidget()
    tab = window.console.tab
    assert tab.isVisible()
    assert tab.parent() is host, 'floated over the views, not a row'
    assert window.console.height() == 0, 'collapsed, the console is no rows'
    assert tab.width() < host.width() / 2, 'a tab, not a bar'
    center_off = abs((tab.x() + tab.width() / 2) - host.width() / 2)
    assert center_off <= 2, f'centered, not parked at an edge ({center_off})'
    assert tab.y() + tab.height() == host.height(), 'riding the bottom edge'
    assert tab.height() == tab.HEIGHT, 'drawn to the design, not styled'
    window.resize(700, 600)
    pump()
    center_off = abs((tab.x() + tab.width() / 2) - host.width() / 2)
    assert center_off <= 2, 'the handle follows the resize'
    window.console.tab.setChecked(True)
    pump()
    assert tab.y() + tab.height() + window.console.height() \
        == host.height(), 'open, the handle rides up on the drawer'


def test_the_tab_expands_and_collapses_the_scrollback(window, pump):
    window.console.tab.setChecked(True)
    pump()
    assert window.console.showing
    window.console.tab.setChecked(False)
    pump()
    assert not window.console.showing


def test_the_scrollback_shows_the_session_as_python(window, pump):
    from visualdynamics.core.averaging import Averaging

    _record(window, pump)
    window.console.tab.setChecked(True)
    window._averaging_edited(Averaging(frame_length=1024, frames=4))
    pump()
    said = window.console.view.toPlainText()
    assert said.startswith('project = visualdynamics.Project(')
    assert "project['Run'].averaging = Averaging(" in said
    assert window.console.view.isReadOnly(), 'copyable, not yet a prompt'


def test_a_settled_drag_rewrites_the_tail_rather_than_stacking(window,
                                                               pump):
    from visualdynamics.core.truncate import Truncation

    _record(window, pump)
    window.console.tab.setChecked(True)
    for stop in (3.0, 2.5, 2.0):
        window._truncation_dragged(Truncation(0.0, stop))
    pump()
    said = window.console.view.toPlainText()
    assert said.count('.truncation = ') == 1, 'one line, the one standing'
    assert 'stop=2.0' in said


def test_a_session_built_object_journals_as_an_honest_comment(window,
                                                              pump):
    """add() takes the object itself, which has no source text; the
    line is a comment the reader sees and exec skips — never a
    plausible call that adds the wrong thing."""
    _record(window, pump)
    said = '\n'.join(window.project.journal)
    assert "# project.add('Run', <TimeHistory built in this session>)" \
        in said
    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    assert 'Run' not in room['project'], 'the comment replayed as nothing'


def test_the_tab_never_covers_the_plots(window, pump):
    """The tab floats over the views' bottom edge and landed on the
    legend row and the bottom axis (Brandon, 2026-09-01). The plot
    area reserves that strip: at least as tall as the tab, all the way
    across, and it survives redraws because it is a layout margin."""
    graphics = window.data_pane.graphics
    tab = window.console.tab
    reserved = graphics.ci.layout.getContentsMargins()[3]
    assert reserved >= tab.sizeHint().height() + 2, (
        f'{reserved} px reserved under {tab.sizeHint().height()} px of tab')
    graphics.clear()
    assert graphics.ci.layout.getContentsMargins()[3] == reserved, (
        'a redraw keeps the clearance')


def test_the_tab_is_translucent_before_it_goes_native(window):
    """The VTK view's `winId()` makes every sibling up the tree native,
    the tab included — and a native child paints on its own surface,
    filled with the window color unless it was declared translucent
    when the surface was made. A gray box around the flared shape on
    a black scene (Brandon, 2026-09-04) is what the attribute
    prevents; it is set in the tab's constructor, ahead of any native
    creation, which is why it is pinned on a fresh tab."""
    from PySide6.QtCore import Qt

    from visualdynamics.gui.drawer_tab import DrawerTab

    fresh = DrawerTab('Console')
    assert fresh.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert window.console.tab.testAttribute(
        Qt.WidgetAttribute.WA_TranslucentBackground)


def test_the_tab_erases_what_was_under_it_before_it_paints(window, pump):
    """The attribute above buys the surface an alpha channel; it does
    not clear it. A native child paints into the top-level's backing
    image at its own place and Qt copies that patch to the child's
    surface, and a translucent widget gets no background erase — so
    once the host's relayout (the console opening, closing) painted
    the window color through the tab's rectangle, every later paint
    drew the tab over a filled box and left the fill. Measured on the
    tab's own CALayer surface, 2026-09-12: filled after one
    `setFixedHeight` on the console, and neither update nor repaint
    took it back. So the tab erases its rectangle to transparent
    before it draws. Pinned by painting the tab over an opaque image:
    the corners outside the flared outline must come out clear."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QColor, QImage, QPainter

    tab = window.console.tab
    tab.resize(tab.sizeHint())
    pump()
    image = QImage(tab.size() * 2, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(2)
    image.fill(QColor(50, 50, 50))
    painter = QPainter(image)
    tab.render(painter, QPoint(), renderFlags=tab.RenderFlag.DrawChildren)
    painter.end()
    w, h = image.width(), image.height()
    for corner in ((2, 2), (w - 3, 2)):
        assert image.pixelColor(*corner).alpha() == 0, (
            f'the corner {corner} kept the fill: {image.pixelColor(*corner).getRgb()}')
    assert image.pixelColor(w // 2, h // 2).alpha() == 255, 'the body is painted'


def test_a_huge_journal_line_is_shown_elided_and_kept_whole(window, pump):
    """A live edit of a controller's target journals a draft of sixty
    thousand band pairs; rendering it twice per drag was the drag's
    whole cost (2026-09-06). The console shows its head; the journal
    and the session script keep it whole."""
    from visualdynamics.gui.console import SHOWN_CHARACTERS, elided

    line = "project.author_specification('Spec', SpecificationDraft(" + \
        'x' * 300_000 + '), replace=True)'
    assert elided('short') == 'short'
    shown = elided(line)
    assert shown.startswith(line[:SHOWN_CHARACTERS])
    assert f'{len(line):,} characters' in shown and 'session script' in shown
    window.project.journal.append(line)
    window._show_status('anything')
    pump()
    text = window.console.view.toPlainText()
    assert len(text) < len(line) // 10
    assert f'{len(line):,} characters' in text
    assert window.project.journal[-1] == line, 'the journal is untouched'
    assert line in window.project.session_script()

