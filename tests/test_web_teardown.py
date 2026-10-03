"""The test helpers' web-view teardown ends the page's render process
before the page goes (2026-10-02): a page torn down with a live
renderer could hang a gate worker for good, by delete or by discard.
"""

from __future__ import annotations

import os

import pytest
from conftest import web_close, web_read, web_view


def test_closing_a_view_ends_its_render_process_first(qt_app):
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    view = web_view()
    view.setHtml('<html><body><p id="here">here</p></body></html>')
    view.show()
    web_read(view, "document.getElementById('here') ? 'yes' : null",
             ready=lambda v: v == 'yes')
    pid = int(view.page().renderProcessPid())
    assert pid > 0, 'a loaded page has a render process'
    web_close(view)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
