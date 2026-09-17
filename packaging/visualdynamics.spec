# PyInstaller spec for Visual Dynamics — one file, all three platforms.
#
#     pyinstaller packaging/visualdynamics.spec --noconfirm
#
# **--onedir, never --onefile**, and that is a license requirement rather
# than a preference. PySide6 is LGPL-3.0, which permits a closed-source
# application only if the recipient can replace the Qt libraries with
# their own build (see NOTICE.md). A directory bundle keeps them as
# separate dylibs/DLLs/.so files and satisfies that; a single
# self-extracting binary does not, and is the one packaging choice that
# would have to be undone later.
#
# The heavy lifting is done by PyInstaller's own hooks for PySide6, VTK
# and numpy — they are maintained upstream and know far more about those
# packages than a hand-written list would. What is here is the rest: the
# entry point, what to leave out, and the platform wrapping.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent          # noqa: F821 — PyInstaller injects it
sys.path.insert(0, str(ROOT))
sys.path.insert(0, SPECPATH)          # noqa: F821
from system_libraries import strip_system_libraries  # noqa: E402
from visualdynamics import __version__  # noqa: E402

MAC = sys.platform == 'darwin'
WINDOWS = sys.platform == 'win32'

# Modules imported lazily inside functions, which the analysis cannot
# see: every io backend is imported inside its own `load`/`save`, and
# the Qt WebEngine pair only when a report is opened.
HIDDEN = [
    'visualdynamics.io.exodus', 'visualdynamics.io.unv',
    'visualdynamics.io.rattlesnake', 'visualdynamics.io.native',
    'visualdynamics.io.excel', 'visualdynamics.io.sdynpy_data',
    'visualdynamics.io.stl', 'visualdynamics.io.threemf',
    'visualdynamics.io.step',
    # the STEP kernel's bindings: one package, many lazily-imported
    # submodules the analysis cannot see through io.step's function
    # bodies
    *collect_submodules('OCP'),
    'visualdynamics.io.sdynpy_npz', 'visualdynamics.io.sdynpy_shapes',
    'PySide6.QtWebEngineWidgets', 'PySide6.QtWebChannel',
    'PySide6.QtOpenGL', 'PySide6.QtSvg',
    'netCDF4', 'h5py', 'openpyxl', 'pandas', 'pint',
]

# Weight that earns nothing in a shipped application: development
# tooling that arrives through some dependency's own extras.
#
# scipy was on this list and had to come off (2026-08-27). It is called
# at runtime — the low-pass, the integration, the filter response curve,
# the sine matched filter — so excluding it built an application that
# raised ImportError on Filter Data, while every test passed on a
# machine that had scipy for the suite's own cross-checks.
# `tests/test_declared_dependencies.py` now holds the two lists to each
# other so the gap cannot reopen.
EXCLUDED = [
    'pytest', 'ruff', 'mypy', 'coverage',
    'mkdocs', 'mkdocstrings', 'griffe', 'markdown',
    'IPython', 'jupyter', 'notebook', 'matplotlib.tests',
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2',
]

datas = [
    # pint ships its unit definitions as data, and every unit string in
    # the application is parsed through them
    *collect_data_files('pint'),
    # the icon the window and the About box draw from is code, not a
    # file — nothing to collect for it; the glyphs the widgets wear
    # are files, read back through importlib.resources
    (str(ROOT / 'src' / 'visualdynamics' / 'gui' / 'icons'),
     'visualdynamics/gui/icons'),
    # The LGPL paper trail, in every package: the third-party notices,
    # the license texts, and the how-to-replace-Qt page. Shipping Qt
    # as replaceable libraries (--onedir) is only half the obligation;
    # the recipient also has to be told, in the package they received.
    (str(ROOT / 'NOTICE.md'), '.'),
    (str(ROOT / 'docs' / 'qt-replacement.md'), '.'),
    (str(ROOT / 'packaging' / 'licenses'), 'licenses'),
]

analysis = Analysis(                                       # noqa: F821
    [str(ROOT / 'packaging' / 'entry.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDED,
    noarchive=False,
    optimize=0,
)

# On Linux the C++ runtime belongs to the host, not to us: mesa's driver
# resolves against whatever `libstdc++` is already loaded, and ours being
# older than the distribution's is what stops the application starting on
# any release newer than the build machine. See system_libraries.py for
# the measurement.
analysis.binaries = strip_system_libraries(analysis.binaries, sys.platform)

pyz = PYZ(analysis.pure)                                   # noqa: F821

executable = EXE(                                          # noqa: F821
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name='Visual Dynamics' if MAC else 'VisualDynamics',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX and code signing do not get along
    console=False,      # a GUI application: no terminal window behind it
    disable_windowed_traceback=False,
    argv_emulation=MAC,  # so a double-clicked .vdyn arrives as argv
    target_arch=None,    # the architecture of the machine building it
    codesign_identity=None,
    entitlements_file=str(ROOT / 'packaging' / 'entitlements.plist')
    if MAC else None,
    icon=str(ROOT / 'packaging' / 'icon.icns') if MAC else
    (str(ROOT / 'packaging' / 'icon.ico') if WINDOWS else None),
)

# Qt learns a Linux desktop's light/dark preference through a
# platform-theme plugin — libqgtk3 on GNOME, libqxdgdesktopportal over
# the settings portal — and PyInstaller's hook collects neither, so the
# AppImage opened light on a dark desktop (a friend of Brandon's,
# 2026-09-14). The wheel ships both; they ride along, and a plugin
# whose libraries the host lacks (gtk3, say) is simply not loaded.
platform_themes = []
if not MAC and not WINDOWS:
    import PySide6
    themes_dir = Path(PySide6.__file__).parent / 'Qt' / 'plugins' / 'platformthemes'
    if themes_dir.is_dir():
        platform_themes = [Tree(str(themes_dir),               # noqa: F821
                               prefix='PySide6/Qt/plugins/platformthemes')]

collected = COLLECT(                                       # noqa: F821
    executable,
    analysis.binaries,
    analysis.datas,
    *platform_themes,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Visual Dynamics' if MAC else 'VisualDynamics',
)

if MAC:
    BUNDLE(                                                # noqa: F821
        collected,
        name='Visual Dynamics.app',
        icon=str(ROOT / 'packaging' / 'icon.icns'),
        bundle_identifier='org.visualdynamics.app',
        version=__version__,
        info_plist={
            'CFBundleShortVersionString': __version__,
            'CFBundleVersion': __version__,
            'NSHighResolutionCapable': True,
            # the project file, so Finder shows the right icon and a
            # double-click opens the application
            'CFBundleDocumentTypes': [{
                'CFBundleTypeName': 'Visual Dynamics Project',
                'CFBundleTypeExtensions': ['vdyn'],
                'CFBundleTypeRole': 'Editor',
                'LSHandlerRank': 'Owner',
            }],
        },
    )
