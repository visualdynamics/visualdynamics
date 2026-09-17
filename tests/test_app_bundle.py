"""The Mac icon that launches the app.

`tools/make_app.py` writes a real `.app` — a bundle Finder, Spotlight and
the Dock treat as an application — whose executable runs this checkout's
own interpreter. Nothing is frozen: the dependencies are two gigabytes of
Qt and VTK, and a bundle carrying them would go stale the moment the
source changed.

The icon is drawn by the package, not shipped, so the Dock tile and the
window's own icon are one drawing.
"""

from __future__ import annotations

import pathlib
import plistlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))


@pytest.fixture(scope='module')
def maker():
    return pytest.importorskip('make_app')


# ---- the drawing --------------------------------------------------------


@pytest.mark.parametrize('size', [16, 32, 64, 128, 256, 512, 1024])
def test_the_icon_draws_at_every_size_it_is_asked_for(qt_app, size):
    from visualdynamics.gui.icons import draw_app_icon

    image = draw_app_icon(size)
    assert image.width() == size and image.height() == size
    assert not image.isNull()


def test_the_icon_has_ink_on_it(qt_app):
    """A tile that renders empty is a tile that would ship empty, and an
    icon nobody can see is not obviously broken from the outside."""
    from visualdynamics.gui.icons import draw_app_icon

    image = draw_app_icon(256)
    opaque = sum(1 for x in range(0, 256, 4) for y in range(0, 256, 4)
                 if image.pixelColor(x, y).alpha() > 0)
    assert opaque > 1000, 'the tile drew'
    middle = image.pixelColor(128, 128)
    corner = image.pixelColor(2, 2)
    assert corner.alpha() == 0, 'the corners are rounded away'
    assert middle.alpha() == 255


def test_it_is_colored_by_amplitude(qt_app):
    """The curves wear viridis by level, which is what the waterfall
    stage colors data with — so the icon is the 3-D reading and not
    a picture of one. Both ends of the map: peaks reach the top,
    antiresonance notches the bottom — the decade clip exists so one
    deep notch cannot flatten every peak into green."""
    from visualdynamics.gui.icons import VIRIDIS, draw_app_icon

    image = draw_app_icon(256)
    seen = {(c.red(), c.green(), c.blue())
            for x in range(0, 256, 2) for y in range(0, 256, 2)
            for c in [image.pixelColor(x, y)] if c.alpha() == 255}

    def near(stop, tolerance=40):
        return any(abs(r - stop[0]) < tolerance and abs(g - stop[1]) < tolerance
                   and abs(b - stop[2]) < tolerance for r, g, b in seen)

    assert near(VIRIDIS[-1]), 'the resonant peaks reach the top of the map'
    assert near(VIRIDIS[0]), 'and the antiresonances the bottom'
    assert near(VIRIDIS[len(VIRIDIS) // 2]), 'with the ramps between'


def test_viridis_runs_dark_to_bright(qt_app):
    from visualdynamics.gui.icons import viridis

    dark, bright = viridis(0.0), viridis(1.0)
    assert dark.lightness() < bright.lightness()
    assert viridis(-5).getRgb() == dark.getRgb(), 'clamped'
    assert viridis(5).getRgb() == bright.getRgb()
    middle = viridis(0.5)
    assert dark.lightness() < middle.lightness() < bright.lightness()


def test_a_small_icon_is_not_the_large_one_shrunk(qt_app):
    """Four ridges over thirty pixels is a smear; the small tile
    carries three."""
    from visualdynamics.gui.app_icon import SMALL, curve_count

    assert curve_count(SMALL) == 4
    assert curve_count(SMALL - 1) == 3
    assert curve_count(32) == 3


def test_the_curves_are_the_plates_own_accelerances(qt_app):
    """The icon's data is synthesized fresh from the demonstration
    plate's eigensolution — real modes, no noise, one source of
    truth. Pinned against an independent synthesis so a drifted
    recipe cannot ship quietly."""
    from visualdynamics.demo import plate
    from visualdynamics.gui.app_icon import (
        BAND,
        DAMPING,
        DECADES,
        DRIVE,
        POINTS,
        RESPONSES,
        synthesized_levels,
    )

    shapes = plate.build().eigensolution(maximum_frequency=2000.0,
                                         damping=DAMPING)
    freq = np.linspace(BAND[0], BAND[1], POINTS)
    rows = np.log10(np.stack([
        np.abs(shapes.synthesize_frf(freq, [dof], [DRIVE], power=2)[0])
        for dof in RESPONSES]))
    expected = np.clip((rows - (rows.max() - DECADES)) / DECADES,
                       0.0, 1.0)
    assert np.allclose(synthesized_levels(), expected)
    assert expected.min() == 0.0 and expected.max() == 1.0, \
        'the clipped range is fully used — peaks and notches both'


def test_the_view_is_the_stages_own(qt_app):
    """The icon's projection basis derives from the same camera
    place_camera gives the waterfall — measured against it, so the
    icon cannot drift from the app's default 3-D view."""
    from visualdynamics.gui.app_icon import stage_basis
    from visualdynamics.viz.waterfall import place_camera

    class Fake:
        camera_position = None
    fake = Fake()
    place_camera(fake)
    position, focus, up = (np.array(v, dtype=float)
                           for v in fake.camera_position)
    view = focus - position
    view /= np.linalg.norm(view)
    expected_right = np.cross(view, up)
    expected_right /= np.linalg.norm(expected_right)
    right, true_up = stage_basis()
    assert np.allclose(right, expected_right)
    assert np.allclose(true_up, np.cross(expected_right, view))


def test_the_window_wears_the_same_icon(qt_app):
    from visualdynamics.gui.icons import app_icon

    icon = app_icon()
    assert not icon.isNull()
    sizes = {(s.width(), s.height()) for s in icon.availableSizes()}
    assert (512, 512) in sizes and (16, 16) in sizes, (
        'several sizes, so a window manager picks rather than scales')


def test_the_app_names_itself(qt_app):
    """Without it macOS names everything after the interpreter."""
    from visualdynamics.gui import main  # noqa: F401  — the module sets it in main()

    source = (ROOT / 'src' / 'visualdynamics' / 'gui' / '__init__.py').read_text(encoding='utf-8')
    assert "setApplicationName('Visual Dynamics')" in source
    assert "setApplicationDisplayName('Visual Dynamics')" in source
    assert 'setWindowIcon(app_icon())' in source


# ---- the bundle ---------------------------------------------------------


@pytest.mark.skipif(not shutil.which('iconutil'),
                    reason='iconutil is macOS\'s own converter')
def test_it_builds_a_bundle_macos_accepts(qt_app, maker, tmp_path):
    app = maker.build(tmp_path)
    assert app == tmp_path / 'Visual Dynamics.app'
    plist = app / 'Contents' / 'Info.plist'
    launcher = app / 'Contents' / 'MacOS' / 'visualdynamics'
    icns = app / 'Contents' / 'Resources' / 'visualdynamics.icns'
    for path in (plist, launcher, icns):
        assert path.exists(), path

    # macOS's own validator, so this is what Finder thinks and not what
    # plistlib was willing to write
    subprocess.run(['plutil', '-lint', str(plist)], check=True,
                   capture_output=True)
    read = plistlib.loads(plist.read_bytes())
    assert read['CFBundleExecutable'] == 'visualdynamics'
    assert read['CFBundleIconFile'] == 'visualdynamics.icns'
    assert read['CFBundlePackageType'] == 'APPL'
    assert read['NSHighResolutionCapable'] is True
    assert read['LSUIElement'] is False, 'a windowed app, not a menu extra'
    assert icns.stat().st_size > 10_000, 'the icns carries every size'


@pytest.mark.skipif(not shutil.which('iconutil'), reason='macOS only')
def test_the_launcher_is_executable_and_points_at_this_checkout(
        qt_app, maker, tmp_path):
    launcher = maker.build(tmp_path) / 'Contents' / 'MacOS' / 'visualdynamics'
    assert launcher.stat().st_mode & 0o111, 'Finder has to be able to run it'
    text = launcher.read_text(encoding='utf-8')
    assert str(ROOT / '.venv' / 'bin' / 'python') in text, (
        'the checkout it was built from, by absolute path — a bundle in '
        '~/Applications has no working directory to lean on')
    assert '-m visualdynamics' in text
    assert 'exec -a visualdynamics' in text, (
        "or every list that names a running process calls it 'python'")


@pytest.mark.skipif(not shutil.which('iconutil'), reason='macOS only')
def test_building_over_an_old_bundle_replaces_it(qt_app, maker, tmp_path):
    app = maker.build(tmp_path)
    (app / 'Contents' / 'stale').write_text('from a previous build')
    again = maker.build(tmp_path)
    assert again == app
    assert not (app / 'Contents' / 'stale').exists()
    assert (app / 'Contents' / 'MacOS' / 'visualdynamics').exists()


@pytest.mark.skipif(not shutil.which('iconutil'), reason='macOS only')
def test_the_launcher_starts_the_app(qt_app, maker, tmp_path):
    """The one thing that matters, run rather than reasoned about.

    Not under the offscreen platform: VTK's render widget cannot be
    created there and takes the process down with it, which is exactly
    why the suite's own windows are built with `offscreen_3d`. So this
    asks the launcher to import and build the window the way it will on
    a real launch, and only skips the event loop.
    """
    launcher = maker.build(tmp_path) / 'Contents' / 'MacOS' / 'visualdynamics'
    probe = (
        'import visualdynamics, sys\n'
        'from visualdynamics.gui.main_window import MainWindow\n'
        'from PySide6.QtWidgets import QApplication\n'
        'app = QApplication.instance() or QApplication([])\n'
        'w = MainWindow(offscreen_3d=True)\n'
        'print("built", visualdynamics.__version__)\n'
    )
    # the launcher with the module swapped for the probe: same
    # interpreter, same path, same everything but the event loop
    python = launcher.read_text(encoding='utf-8').split('"')[1]
    result = subprocess.run([python, '-c', probe], capture_output=True,
                            text=True, timeout=300, check=False,
                            env={'PATH': '/usr/bin:/bin',
                                 'QT_QPA_PLATFORM': 'offscreen',
                                 'HOME': str(tmp_path)})
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'built' in result.stdout


def test_the_packages_carry_the_lgpl_paper_trail():
    """--onedir makes Qt replaceable; the paper half of the LGPL is
    that the recipient is *told* — the notices, the license texts and
    the replacement instructions must travel inside every package.
    The spec's datas list is where that is or is not true."""
    import pathlib

    root = pathlib.Path(__file__).parent.parent
    spec = (root / 'packaging' / 'visualdynamics.spec'
            ).read_text(encoding='utf-8')
    # the datas tuples, not the filename anywhere: a comment saying
    # "see NOTICE.md" satisfied the weaker check while shipping nothing
    for entry in ("(str(ROOT / 'NOTICE.md'), '.')",
                  "(str(ROOT / 'docs' / 'qt-replacement.md'), '.')",
                  "(str(ROOT / 'packaging' / 'licenses'), 'licenses')"):
        assert entry in spec, f'{entry} is not in the datas — nothing ships'
    for text in ('LGPL-3.0.txt', 'GPL-3.0.txt'):
        assert (root / 'packaging' / 'licenses' / text).exists(), (
            f'{text} is missing — LGPLv3 incorporates the GPL by '
            'reference, so both texts ship')
    # the instructions exist, cover every platform, and the notice
    # points at them
    page = (root / 'docs' / 'qt-replacement.md'
            ).read_text(encoding='utf-8')
    for platform in ('macOS', 'Windows', 'Linux'):
        assert f'## {platform}' in page
    assert 'codesign' in page, (
        'without the re-sign step, replacement on Apple silicon is a '
        'right in name only')
    notice = (root / 'NOTICE.md').read_text(encoding='utf-8')
    assert 'qt-replacement.md' in notice


def test_the_linux_package_ships_the_platform_theme_plugins():
    """Qt learns a Linux desktop's light/dark preference through a
    platform-theme plugin, and PyInstaller's hook collects none, so
    the AppImage opened light on a dark desktop (a friend of Brandon's,
    2026-09-14). The spec adds the wheel's platformthemes directory to
    the collection on Linux; pinned on the spec's own words, since the
    Linux build runs elsewhere."""
    import pathlib

    root = pathlib.Path(__file__).parent.parent
    spec = (root / 'packaging' / 'visualdynamics.spec'
            ).read_text(encoding='utf-8')
    assert "'platformthemes'" in spec and "prefix='PySide6/Qt/plugins/platformthemes'" in spec
    assert '*platform_themes,' in spec, 'built, but not handed to COLLECT'
