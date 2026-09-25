"""Copying a report figure to the clipboard.

Every figure on the report page carries a copy button in its corner,
shown on hover the way a chat's code blocks show theirs (Brandon,
2026-09-24), so a plot goes into a document or an email without a
screenshot. The canvas is copied as drawn, at its own resolution.

One button, two ways off the page, and both are tested here: inside
the application the PNG crosses the web channel to `_Bridge.copy_image`,
which owns the real clipboard; the exported file, open in a browser,
uses the browser's clipboard API. The button must not print, and in
edit mode its click must not select the block it sits on.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from conftest import web_close, web_read

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.report import Report
from visualdynamics.report import render_html


def _png_data_url(width, height):
    """A real PNG of that size, as the page's FileReader would send it."""
    import base64

    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QColor, QImage

    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor('orange'))
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, 'PNG')
    encoded = base64.b64encode(bytes(buffer.data())).decode('ascii')
    return 'data:image/png;base64,' + encoded


def _report(tmp_path=None):
    """Two figures, a text block and, given somewhere to write a
    photograph, a table and a photo as well."""
    fs = 1024.0
    t = np.arange(2048) / fs
    history = TimeHistory(
        t, np.random.default_rng(3).standard_normal((2, len(t))),
        response_dof=['101Z+', '104Z+'], ordinate_dim='acceleration')
    blocks = [
        {'kind': 'text', 'text': 'lead-in'},
        {'kind': 'plot', 'source': 'Time History', 'mode': 'time',
         'select': 'dim:acceleration', 'caption': 'The recording'},
        {'kind': 'plot', 'source': 'Time History', 'mode': 'scalogram',
         'select': 'dim:acceleration', 'caption': 'Scalogram'}]
    objects = {'Time History': history}
    if tmp_path is not None:
        from PySide6.QtGui import QImage

        from visualdynamics.core.photos import Photos
        from visualdynamics.core.shapes import ShapeSet

        objects['Modes'] = ShapeSet([10.0, 25.0], [0.01, 0.02],
                                    ['101Z+', '104Z+'],
                                    [[1.0, 0.5], [0.5, -1.0]])
        picture = tmp_path / 'Setup.png'
        QImage(12, 9, QImage.Format.Format_RGB32).save(str(picture))
        album = Photos()
        album.add_file(str(picture))
        objects['Photos'] = album
        blocks += [
            {'kind': 'table', 'source': 'Modes', 'caption': 'The modes'},
            {'kind': 'photo', 'source': 'Photos', 'photo': 'Setup',
             'caption': 'The setup'}]
    return Report('R', blocks), objects


def _view(path):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(['x'])
    view = QWebEngineView()
    view.resize(1000, 800)
    view.load(QUrl.fromLocalFile(str(path)))
    view.show()
    return view


#: the page is one inline script: complete means it ran to the end or
#: threw, and a thrown page has no copy buttons to find
READY = ("if (document.readyState !== 'complete'"
         "    || !document.getElementById('data')"
         "    || !document.querySelector('.copy')) return null;")


def test_every_figure_has_a_copy_button_hidden_until_hovered_and_never_printed(
        tmp_path):
    report, objects = _report()
    path = tmp_path / 'read.html'
    path.write_text(render_html(report, objects), encoding='utf-8')
    view = _view(path)
    try:
        answer = web_read(
            view,
            "(() => {" + READY +
            "const printHides = Array.from(document.styleSheets)"
            "  .flatMap(sheet => Array.from(sheet.cssRules))"
            "  .filter(rule => rule.media && rule.media.mediaText === 'print')"
            "  .flatMap(rule => Array.from(rule.cssRules))"
            "  .some(rule => rule.selectorText === '.copy'"
            "                && rule.style.display === 'none');"
            "return JSON.stringify({"
            " canvases: document.querySelectorAll('section canvas').length,"
            " buttons: document.querySelectorAll('.copy').length,"
            " besideACanvas: Array.from(document.querySelectorAll('.copy'),"
            "   b => b.previousElementSibling"
            "        && b.previousElementSibling.tagName === 'CANVAS').every(Boolean),"
            " textHasNone: document.querySelectorAll('section')[0]"
            "   .querySelector('.copy') === null,"
            " hidden: getComputedStyle(document.querySelector('.copy')).opacity,"
            " title: document.querySelector('.copy').title,"
            " printHides: printHides}); })()")
    finally:
        web_close(view)
    found = json.loads(answer)
    assert found['canvases'] == 2
    assert found['buttons'] == 2, 'one button per figure, none elsewhere'
    assert found['besideACanvas'], 'each sits over its own canvas'
    assert found['textHasNone']
    assert found['hidden'] == '0', 'shown on hover only'
    assert found['title'] == 'Copy this figure as an image'
    assert found['printHides'], 'a PDF is not a page to copy from'


def test_tables_and_photos_carry_the_button_too(tmp_path):
    """The rest of the request: every table and every photograph on the
    page has the same button, hung where it reads — a photo's on a
    holder its own width, a table's on the section rather than the
    wrap that scrolls a wide table sideways."""
    report, objects = _report(tmp_path)
    path = tmp_path / 'read.html'
    path.write_text(render_html(report, objects), encoding='utf-8')
    view = _view(path)
    try:
        answer = web_read(
            view,
            "(() => {" + READY +
            "const buttons = Array.from(document.querySelectorAll('.copy'));"
            "return JSON.stringify({"
            " canvases: document.querySelectorAll('section canvas').length,"
            " tables: document.querySelectorAll('section table').length,"
            " photos: document.querySelectorAll('.photoholder img').length,"
            " buttons: buttons.length,"
            " titles: buttons.map(b => b.title),"
            " photoHolder: !!document.querySelector('.photoholder > .copy'),"
            " tableOnSection: !!document.querySelector('section > .copy')"
            "   && !document.querySelector('.tablewrap > .copy')}); })()")
    finally:
        web_close(view)
    found = json.loads(answer)
    assert (found['canvases'], found['tables'], found['photos']) == (2, 1, 1)
    assert found['buttons'] == 4, 'one per figure, table and photo'
    assert sorted(found['titles']) == sorted([
        'Copy this figure as an image', 'Copy this figure as an image',
        'Copy this table as cells and as a table', 'Copy this photo'])
    assert found['photoHolder'] and found['tableOnSection']


def test_a_table_and_a_photo_cross_the_bridge_in_their_own_forms(tmp_path):
    """A table goes as tab-separated text and as HTML; a photo as a
    PNG, through the same slot a figure uses."""
    report, objects = _report(tmp_path)
    html = render_html(
        report, objects, None, edit=True,
        channel_js='window.__sent = []; window.__table = null;'
        ' window.__image = null;'
        'window.QWebChannel = function(t, cb) { cb({objects: {bridge: {'
        '  apply: p => window.__sent.push(JSON.parse(p)),'
        '  copy_image: (url, answer) => { window.__image = url.slice(0, 22);'
        '                                 answer(true); },'
        '  copy_table: (text, markup, answer) => {'
        '    window.__table = {text, markup}; answer(true); }}}}); };'
        'window.qt = {webChannelTransport: {}};')
    path = tmp_path / 'edit.html'
    path.write_text(html, encoding='utf-8')
    view = _view(path)
    try:
        answer = web_read(
            view,
            "(() => {" + READY +
            "if (!window.__bridge) return null;"
            "const table = document.querySelector('section > .copy"
            "  + .caption, section > .tablewrap ~ .copy')"
            "  || Array.from(document.querySelectorAll('.copy'))"
            "     .find(b => b.title.startsWith('Copy this table'));"
            "const photo = document.querySelector('.photoholder > .copy');"
            "if (!window.__clicked) { window.__clicked = true;"
            "  table.click(); photo.click(); return null; }"
            "if (!window.__table || !window.__image"
            "    || !table.dataset.last || !photo.dataset.last) return null;"
            "return JSON.stringify({table: window.__table,"
            "  image: window.__image,"
            "  last: [table.dataset.last, photo.dataset.last],"
            "  selected: document.querySelectorAll('section.selected').length"
            "}); })()")
    finally:
        web_close(view)
    found = json.loads(answer)
    lines = found['table']['text'].split('\n')
    assert lines[0].split('\t') == ['Mode', 'Frequency [Hz]', 'Damping [%]',
                                    'Description']
    assert len(lines) == 3 and '\t' in lines[1], 'a header and two modes'
    assert found['table']['markup'].startswith('<table')
    assert found['image'] == 'data:image/png;base64,'
    assert found['last'] == ['copied', 'copied']
    assert found['selected'] == 0


def test_the_bridge_puts_a_table_on_the_clipboard_in_both_forms(qt_app):
    from PySide6.QtWidgets import QApplication

    from visualdynamics.gui.report_editor import _Bridge

    bridge = _Bridge()
    assert bridge.copy_table('a\tb\n1\t2', '<table><tr><td>a</td></tr></table>')
    mime = QApplication.clipboard().mimeData()
    assert mime.text() == 'a\tb\n1\t2'
    assert mime.hasHtml() and '<table>' in mime.html()
    assert bridge.copy_table('   ', '<table></table>') is False


def test_in_a_browser_the_figure_goes_to_the_browsers_clipboard_as_png(
        tmp_path):
    """The exported file has no bridge; the button writes a PNG through
    the page's own clipboard API, as an item, which is what the
    browsers that take images want."""
    report, objects = _report()
    path = tmp_path / 'read.html'
    path.write_text(render_html(report, objects), encoding='utf-8')
    view = _view(path)
    try:
        answer = web_read(
            view,
            "(() => {" + READY +
            "if (!window.__clicked) {"
            "  window.__clicked = true; window.__wrote = null;"
            "  Object.defineProperty(navigator, 'clipboard', {"
            "    configurable: true, value: {write: items => {"
            "      window.__wrote = items.map(i => ({"
            "        item: i instanceof ClipboardItem, types: i.types}));"
            "      return Promise.resolve(); }}});"
            "  document.querySelector('.copy').click();"
            "  return null; }"
            "const button = document.querySelector('.copy');"
            "if (window.__wrote === null || !button.dataset.last)"
            "  return null;"
            "return JSON.stringify({wrote: window.__wrote,"
            "  last: button.dataset.last}); })()")
    finally:
        web_close(view)
    found = json.loads(answer)
    assert found['wrote'] == [{'item': True, 'types': ['image/png']}]
    assert found['last'] == 'copied', 'and the button says so'


def test_in_the_application_the_figure_crosses_the_bridge_and_selects_nothing(
        tmp_path):
    """Edit mode: the bytes go to the bridge's `copy_image` as a PNG data
    URL, the answer decides what the button says, and the click stays on
    the button — a click on a section selects the block, and copying a
    figure is not choosing it."""
    report, objects = _report()
    html = render_html(
        report, objects, None, edit=True,
        channel_js='window.__sent = []; window.__copied = null;'
        'window.QWebChannel = function(t, cb) { cb({objects: {bridge: {'
        '  apply: p => window.__sent.push(JSON.parse(p)),'
        '  copy_image: (url, answer) => { window.__copied = url.slice(0, 22);'
        '                                 answer(true); }}}}); };'
        'window.qt = {webChannelTransport: {}};')
    path = tmp_path / 'edit.html'
    path.write_text(html, encoding='utf-8')
    view = _view(path)
    try:
        answer = web_read(
            view,
            "(() => {" + READY +
            "if (!window.__bridge) return null;"
            "if (!window.__clicked) { window.__clicked = true;"
            "  document.querySelector('.copy').click(); return null; }"
            "const button = document.querySelector('.copy');"
            "if (window.__copied === null || !button.dataset.last)"
            "  return null;"
            "return JSON.stringify({copied: window.__copied,"
            "  last: button.dataset.last,"
            "  selected: document.querySelectorAll('section.selected').length,"
            "  sent: window.__sent}); })()")
    finally:
        web_close(view)
    found = json.loads(answer)
    assert found['copied'] == 'data:image/png;base64,'
    assert found['last'] == 'copied', 'the bridge said yes and the button shows it'
    assert found['selected'] == 0 and found['sent'] == [], (
        'the copy click did not select the block it sits on')


def test_the_bridge_puts_the_image_on_the_clipboard_and_says_so(qt_app):
    from PySide6.QtWidgets import QApplication

    from visualdynamics.gui.report_editor import _Bridge

    QApplication.clipboard().clear()
    bridge = _Bridge()
    assert bridge.copy_image(_png_data_url(7, 3)) is True
    image = QApplication.clipboard().image()
    assert (image.width(), image.height()) == (7, 3)
    assert bridge.copy_image('data:image/png;base64,bm90IGEgcG5n') is False, (
        'bytes that are not an image are refused, so the button can say '
        'it could not copy'
    )


def test_end_to_end_the_pages_button_fills_the_clipboard(window, pump):
    """The real page in the real editor, through the real web channel:
    a click on the button leaves the figure on the clipboard at the
    canvas's own resolution."""
    from PySide6.QtWidgets import QApplication

    report, objects = _report()
    QApplication.clipboard().clear()
    window.add_object('Time History', objects['Time History'])
    window.add_object('Report', report)
    window.tree.setCurrentItem(window._item_for_object('Report'))
    window.render_current()
    pump()
    editor = window.report_editor
    # Click once, whatever the asking: `web_read` re-asks a page whose
    # answer it counts lost after 10 s, and on a loaded CI runner the
    # first asking clicked while the page was still settling (the chart
    # 767 px wide, a scrollbar up), the second clicked again at 785 px
    # and answered that — while the clipboard already held the first
    # copy (2026-09-25). So the size is the canvas's at the click, kept
    # on the window, and the clicks are counted.
    expression = (
        "(() => {" + READY +
        "if (!window.__bridge || !window.__bridge.copy_image) return null;"
        "if (!window.__clicks) {"
        "  const canvas = document.querySelector('section canvas');"
        "  window.__size = {width: canvas.width, height: canvas.height};"
        "  window.__clicks = 1;"
        "  document.querySelector('.copy').click(); }"
        "return JSON.stringify({size: window.__size,"
        "                       clicks: window.__clicks}); })()")
    answer = web_read(editor.view, expression)
    size = json.loads(answer)['size']
    import time

    deadline = time.monotonic() + 30.0
    image = QApplication.clipboard().image()
    while image.isNull() and time.monotonic() < deadline:
        pump()
        image = QApplication.clipboard().image()
    assert not image.isNull(), 'the page put an image on the clipboard'
    assert (image.width(), image.height()) == (size['width'], size['height']), (
        'the copy is the canvas at its own backing-store resolution'
    )
    # asked again, as a lost answer makes web_read do, the page clicks
    # no second time
    assert json.loads(web_read(editor.view, expression))['clicks'] == 1, (
        'a re-asked page copied twice'
    )
    # and the bridge's answer reached the button: a slot with no return
    # value fills the clipboard and leaves the button saying it failed
    last = web_read(editor.view,
                    "document.querySelector('.copy').dataset.last || null")
    assert last == 'copied'
