"""Print a rendered report page to a paginated PDF.

The PDF is the report page itself, printed by the web engine the
report editor already uses (Qt WebEngine's `printToPdf`), so what is
issued is what the page shows. The page is loaded with `#print`, which
draws its figures at print resolution, and its print stylesheet keeps a
figure off a page break. With the report feature `report_identity` on,
the page's margins carry the laboratory, the report's number and
revision, the title and page n of N, numbered by the print engine's own
page counters.

The printing runs in the web engine's processes while the
application's thread turns the event loop, so a window stays painted.
Called from a worker thread, the print is handed to the application's
thread and the worker waits for it. It needs a Qt application, and
makes one when a script has none.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any

#: the page sizes a report prints on
PAGE_SIZES = ('letter', 'a4')

#: the width a page is laid out at before it is printed, in CSS pixels:
#: a letter page's printable width between the identity margins, so
#: each figure is drawn at the width it prints at
PRINT_WIDTH = 700


def _application() -> Any:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        # before the application exists, or the web engine refuses to
        # share GL contexts (`gui/__init__.py`)
        QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        app = QApplication([])
    return app


def _wait(app: Any, done: Any, timeout: float) -> bool:
    from PySide6.QtCore import QEventLoop

    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        # the window stays painted; a click waits until the print is
        # done, as it does for any long verb (`MainWindow._run_long`)
        app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        time.sleep(0.005)
    return bool(done())


def _on_gui_thread(app: Any, work: Any) -> Any:
    """Run `work` on the application's thread and return what it
    returns (or raise what it raises), with this thread waiting. The
    web engine lives on that thread, and a long verb runs on a worker
    while that thread turns its event loop, which delivers the call."""
    from PySide6.QtCore import QObject, Qt, Signal

    class Courier(QObject):
        asked = Signal()

    outcome: list[tuple[bool, Any]] = []

    def run() -> None:
        try:
            outcome.append((True, work()))
        except BaseException as failure:  # noqa: BLE001 - raised below
            outcome.append((False, failure))

    # a function connected with no receiver runs in the sender's thread,
    # so the courier is moved to the application's
    courier = Courier()
    courier.moveToThread(app.thread())
    courier.asked.connect(run, Qt.ConnectionType.BlockingQueuedConnection)
    courier.asked.emit()
    ok, value = outcome[0]
    if not ok:
        raise value
    return value


def print_pdf(page: str, path: str | os.PathLike, *,
              page_size: str = 'letter', timeout: float = 300.0) -> bytes:
    """Print `page` (a report rendered by `render_html`) to a PDF at
    `path`, and return the PDF's bytes.

    `page_size` is 'letter' or 'a4'. The PDF is printed beside the page
    in a temporary folder and moved to `path` only once it is whole, so
    a failed print leaves `path` as it was.

    Raises
    ------
    ValueError
        For a page size that is not one of `PAGE_SIZES`.
    OSError
        When the page does not load, or the print fails or does not
        finish within `timeout` seconds.
    """
    if page_size not in PAGE_SIZES:
        raise ValueError(f'unknown page size {page_size!r}; the sizes are: '
                         f'{", ".join(PAGE_SIZES)}')
    from PySide6.QtCore import QMarginsF, QUrl
    from PySide6.QtGui import QPageLayout, QPageSize
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView

    app = _application()
    if not app.thread().isCurrentThread():
        return _on_gui_thread(app, lambda: print_pdf(
            page, path, page_size=page_size, timeout=timeout))
    target = os.path.expanduser(str(path))
    size = (QPageSize.PageSizeId.Letter if page_size == 'letter'
            else QPageSize.PageSizeId.A4)
    # the margins the identity rides in (`identity.MARGINS`, which the
    # page's own print stylesheet also states, for a page printed from
    # a browser)
    from .identity import MARGINS

    top, side = MARGINS
    layout = QPageLayout(QPageSize(size), QPageLayout.Orientation.Portrait,
                         QMarginsF(side, top, side, top),
                         QPageLayout.Unit.Inch)
    with tempfile.TemporaryDirectory(prefix='vd-print-') as folder:
        source = os.path.join(folder, 'report.html')
        printed = os.path.join(folder, 'report.pdf')
        with open(source, 'w', encoding='utf-8') as out:
            out.write(page)
        view = QWebEngineView()
        web = QWebEnginePage()
        view.setPage(web)
        view.resize(PRINT_WIDTH, 900)
        state: dict[str, Any] = {}
        web.loadFinished.connect(lambda ok: state.update(loaded=ok))
        web.pdfPrintingFinished.connect(
            lambda _file, ok: state.update(printed=ok))
        url = QUrl.fromLocalFile(source)
        url.setFragment('print')
        try:
            view.load(url)
            if not _wait(app, lambda: 'loaded' in state, timeout) \
                    or not state['loaded']:
                raise OSError(f'the report page did not load within '
                              f'{timeout:g} s, so nothing was printed')
            web.printToPdf(printed, layout)
            if not _wait(app, lambda: 'printed' in state, timeout):
                raise OSError(f'the PDF was not finished within {timeout:g} '
                              's; nothing was written')
            if not state['printed'] or not os.path.isfile(printed):
                raise OSError(f'the web engine could not print {target}')
            with open(printed, 'rb') as handle:
                data = handle.read()
            shutil.move(printed, target)
        finally:
            _stand_down(app, view, web)
    return data


def _stand_down(app: Any, view: Any, web: Any) -> None:
    """Tear the print view down the way the report editor does: stop,
    hide, detach the page, let the widget go, then discard the page, so
    nothing is left for a destructor to wait on (`ReportEditor.stand_down`)."""
    import contextlib

    from PySide6.QtWebEngineCore import QWebEnginePage

    view.stop()
    view.hide()
    view.setPage(None)
    for _ in range(20):
        app.processEvents()
    with contextlib.suppress(RuntimeError):
        web.setLifecycleState(QWebEnginePage.LifecycleState.Discarded)
    for _ in range(20):
        app.processEvents()
    web.deleteLater()
    view.deleteLater()
    for _ in range(10):
        app.processEvents()
