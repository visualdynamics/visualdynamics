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


def _report():
    fs = 1024.0
    t = np.arange(2048) / fs
    history = TimeHistory(
        t, np.random.default_rng(3).standard_normal((2, len(t))),
        response_dof=['101Z+', '104Z+'], ordinate_dim='acceleration')
    report = Report('R', [
        {'kind': 'text', 'text': 'lead-in'},
        {'kind': 'plot', 'source': 'Time History', 'mode': 'time',
         'select': 'dim:acceleration', 'caption': 'The recording'},
        {'kind': 'plot', 'source': 'Time History', 'mode': 'scalogram',
         'select': 'dim:acceleration', 'caption': 'Scalogram'}])
    return report, {'Time History': history}


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
    answer = web_read(
        editor.view,
        "(() => {" + READY +
        "if (!window.__bridge || !window.__bridge.copy_image) return null;"
        "const canvas = document.querySelector('section canvas');"
        "document.querySelector('.copy').click();"
        "return JSON.stringify({width: canvas.width,"
        "                       height: canvas.height}); })()")
    size = json.loads(answer)
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
    # and the bridge's answer reached the button: a slot with no return
    # value fills the clipboard and leaves the button saying it failed
    last = web_read(editor.view,
                    "document.querySelector('.copy').dataset.last || null")
    assert last == 'copied'
