"""The packaged app starts in seconds, not in a font scan (2026-09-28).

It took 15.6 s to show a window where a checkout took 1.6: pyvista
imports matplotlib.pyplot, pyplot loads its font cache, and
PyInstaller's matplotlib hook points MPLCONFIGDIR at a new temporary
folder every launch, so the cache — `system_profiler SPFontsDataType`
on macOS, ten seconds and more — was rebuilt every time.
`packaging/entry.py` points it at a folder that lasts, before anything
imports matplotlib.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

ENTRY = pathlib.Path(__file__).resolve().parent.parent / 'packaging' / 'entry.py'


def _entry():
    spec = importlib.util.spec_from_file_location('vd_entry', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_font_cache_lives_in_a_lasting_per_user_folder(monkeypatch):
    entry = _entry()
    monkeypatch.setattr('sys.platform', 'darwin')
    path = entry.matplotlib_cache()
    assert path.endswith('Library/Caches/org.visualdynamics.app/matplotlib')
    monkeypatch.setattr('sys.platform', 'win32')
    monkeypatch.setenv('LOCALAPPDATA', r'C:\Users\x\AppData\Local')
    assert entry.matplotlib_cache().startswith(r'C:\Users\x\AppData\Local')
    monkeypatch.setattr('sys.platform', 'linux')
    monkeypatch.setenv('XDG_CACHE_HOME', '/home/x/.cache')
    assert entry.matplotlib_cache() == '/home/x/.cache/visualdynamics/matplotlib'
    for path in (entry.matplotlib_cache(),):
        assert 'tmp' not in path.lower()


def test_it_is_set_before_anything_can_import_matplotlib():
    """The whole fix is ordering: the variable after the first import of
    the package is read by nobody, since importing visualdynamics
    imports pyvista, which imports pyplot."""
    tree = ast.parse(ENTRY.read_text(encoding='utf-8'))
    main_block = next(node for node in tree.body if isinstance(node, ast.If))
    order = []
    for node in ast.walk(main_block):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == 'MPLCONFIGDIR'
                and isinstance(node.ctx, ast.Store)):
            order.append(('set', node.lineno))
        if isinstance(node, ast.ImportFrom) and node.module == 'visualdynamics.gui':
            order.append(('import', node.lineno))
    kinds = [kind for kind, _line in sorted(order, key=lambda o: o[1])]
    assert kinds == ['set', 'import'], order
