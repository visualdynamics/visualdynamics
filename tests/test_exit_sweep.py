"""Nothing of Python's left for PySide's exit sweep (`gui.release_widgets`).

Left to the interpreter's exit, PySide deletes every widget it still
owns in one sweep, in no order of its own; a dialog deleted there takes
its children with it, and the sweep then deletes a child again. Dialogs
kept alive only by a reference cycle are the case that crashes: thirty
of them segfaulted the process on every run measured, and none did with
`release_widgets` called first. Each case runs in a process of its own,
since the crash is the process's exit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent('''
    import os, sys, tempfile
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    sys.path.insert(0, {tests!r})
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    from test_large_records import _stream
    from visualdynamics.gui import release_widgets
    from visualdynamics.gui.stream_window import StreamWindowDialog
    from visualdynamics.io import rattlesnake
    from visualdynamics.theme import theme
    path = _stream(os.path.join(tempfile.mkdtemp(), 'run.nc4'), samples=1 << 12)
    summary = rattlesnake.stream_summary(path)
    # held by nothing but their own cycles, as a test's dialog is once
    # the test returns
    for _ in range(30):
        StreamWindowDialog(path, summary, theme('light'))
    print('released', release_widgets(app), flush=True)
''')


def _run():
    tests = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run([sys.executable, '-c', SCRIPT.format(tests=tests)],
                          check=False, capture_output=True, text=True, timeout=120)


def test_dialogs_left_alive_do_not_crash_the_exit():
    for _ in range(2):
        done = _run()
        assert done.returncode == 0, (
            f'exit {done.returncode}: the sweep deleted a widget twice\n{done.stderr[-2000:]}')
        # each dialog and its plot's two context menus, parentless all
        assert int(done.stdout.split()[-1]) >= 30, done.stdout
