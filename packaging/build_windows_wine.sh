#!/bin/zsh
# The Windows installer, built on this Mac under Wine — no Windows
# machine, no license, no Actions minutes.
#
#     packaging/build_windows_wine.sh setup    # once: Wine, Python, tools
#     packaging/build_windows_wine.sh build    # the installer, into dist/
#     packaging/build_windows_wine.sh run      # launch what was built
#
# What this is for: producing the x64 installer and *debugging the
# Windows build locally* — the app runs under Wine well enough to see
# real Windows-only failures (it caught a missing-plugin bundle and a
# DLL chain the first day). What it is not: a substitute for the real
# thing. Ship only after the release workflow's Windows job has built
# and smoked it on actual Windows.
#
# The traps this script exists to remember (found 2026-08-31):
#
# - Wine **devel** (11.16+), not stable (11.0): Qt 6.7+ links the
#   icuuc/icuin DLLs Windows ships natively, and Wine only grew them
#   after the 11.0 cut. Without them QtCore does not load.
# - The prefix needs the VC++ runtime (`winetricks vcrun2022`) —
#   msvcp140.dll is nowhere in a bare prefix and Qt needs it.
# - Windows Python comes from the **NuGet zip**: the python.org
#   installer is a Burn bundle and silently does nothing under Wine.
# - **Build only in a prefix where Qt imports.** PyInstaller's Qt hooks
#   learn the plugin list by importing Qt; in a broken prefix the build
#   exits 0 and bundles zero Qt plugins. The build here refuses to
#   start until `from PySide6 import QtCore` works.
# - VTK needs Mesa's software GL (`opengl32.dll`, llvmpipe) to *run*
#   under Wine — the Mac GL that Wine exposes stops at 2.1 and VTK
#   crashes native. That is a Wine-only need: the DLLs are dropped
#   beside the app for `run`, never into the installer.
# - Only x64. VTK publishes no win_arm64 wheels, so a Windows-on-ARM
#   package is not buildable from any host today.
set -e

WINE_APP="$HOME/Applications/Wine Devel.app"
WINE="$WINE_APP/Contents/Resources/wine/bin/wine"
export WINEPREFIX="$HOME/.wine-visualdynamics"
export WINEDEBUG=-all
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WINREPO="Z:${REPO//\//\\}"
PY='C:\Python313\python.exe'
ISCC='C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
MESA_DLLS=(opengl32.dll libgallium_wgl.dll)

WINE_URL=https://github.com/Gcenx/macOS_Wine_builds/releases/download/11.16/wine-devel-11.16-osx64.tar.xz
PYTHON_NUPKG=https://globalcdn.nuget.org/packages/python.3.13.7.nupkg
INNO_URL=https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe
MESA_URL=https://github.com/pal1000/mesa-dist-win/releases/download/26.2.0/mesa3d-26.2.0-release-msvc.7z

setup() {
    work=$(mktemp -d)
    if [[ ! -x "$WINE" ]]; then
        echo '== Wine devel (stable lacks the ICU DLLs Qt 6.7+ needs)'
        curl -sL -o "$work/wine.tar.xz" "$WINE_URL"
        mkdir -p ~/Applications
        tar -xf "$work/wine.tar.xz" -C ~/Applications
        xattr -dr com.apple.quarantine "$WINE_APP" 2>/dev/null || true
    fi
    "$WINE" wineboot -i 2>/dev/null

    if [[ ! -e "$WINEPREFIX/drive_c/windows/system32/msvcp140.dll" ]]; then
        echo '== VC++ runtime (Qt needs msvcp140; a bare prefix has none)'
        command -v winetricks >/dev/null || brew install winetricks
        WINE="$WINE" PATH="$(dirname "$WINE"):$PATH" winetricks -q vcrun2022
    fi

    if [[ ! -e "$WINEPREFIX/drive_c/Python313/python.exe" ]]; then
        echo '== Windows Python (NuGet zip: the real installer no-ops here)'
        curl -sL -o "$work/python.nupkg" "$PYTHON_NUPKG"
        ( cd "$work" && unzip -oq python.nupkg tools/'*' )
        cp -R "$work/tools" "$WINEPREFIX/drive_c/Python313"
    fi

    echo '== the project and its build tools, into Windows Python'
    "$WINE" "$PY" -m pip install --prefer-binary -q \
        "$WINREPO" cadquery-ocp pyinstaller pillow 2>/dev/null

    if [[ ! -e "$WINEPREFIX/drive_c/Program Files (x86)/Inno Setup 6/ISCC.exe" ]]; then
        echo '== Inno Setup'
        curl -sL -o "$work/innosetup.exe" "$INNO_URL"
        "$WINE" "$work/innosetup.exe" /VERYSILENT /SUPPRESSMSGBOXES \
            /NORESTART 2>/dev/null
    fi

    if [[ ! -e "$WINEPREFIX/drive_c/mesa/opengl32.dll" ]]; then
        echo '== Mesa software GL (VTK needs GL 3.2+; Wine-on-Mac has 2.1)'
        command -v 7z >/dev/null || brew install p7zip
        curl -sL -o "$work/mesa.7z" "$MESA_URL"
        ( cd "$work" && 7z x -y -omesa mesa.7z > /dev/null )
        mkdir -p "$WINEPREFIX/drive_c/mesa"
        for dll in $MESA_DLLS; do
            cp "$work/mesa/x64/$dll" "$WINEPREFIX/drive_c/mesa/"
        done
    fi
    rm -rf "$work"
    echo 'setup complete — packaging/build_windows_wine.sh build'
}

check_qt() {
    # a broken Qt makes a *quietly* broken bundle, so it is a hard stop
    "$WINE" "$PY" -c 'from PySide6 import QtCore' 2>/dev/null || {
        echo 'QtCore does not import under Wine — run setup first;' \
             'building now would bundle no Qt plugins and exit 0' >&2
        exit 1
    }
}

build() {
    check_qt
    cd "$REPO"
    # The project goes into Windows Python as a *copy* (setup installs
    # it once), and PyInstaller freezes whatever that copy is — so a
    # build that does not reinstall ships setup day's code under setup
    # day's version. The 0.1.0 refresh produced a 0.0.1 installer that
    # way (2026-09-01). Reinstall every build, and refuse to freeze a
    # version that is not the tree's.
    "$WINE" "$PY" -m pip install --no-deps -q "$WINREPO" 2>/dev/null
    version=$("$WINE" "$PY" -c \
        'import visualdynamics; print(visualdynamics.__version__)' \
        2>/dev/null | tr -d '\r')
    tree=$(sed -n "s/^__version__ = '\(.*\)'/\1/p" src/visualdynamics/__init__.py)
    [[ "$version" == "$tree" ]] || {
        echo "Windows Python has visualdynamics $version, the tree is" \
             "$tree — the reinstall did not take; not building" >&2
        exit 1
    }
    rm -rf build dist
    "$WINE" "$PY" -m PyInstaller 'packaging\visualdynamics.spec' \
        --noconfirm --distpath dist --workpath build 2>/dev/null
    [[ -e dist/VisualDynamics/_internal/PySide6/plugins/platforms/qwindows.dll ]] || {
        echo 'built, but qwindows.dll is not in the bundle — the Qt' \
             'hooks failed; do not ship this' >&2
        exit 1
    }
    "$WINE" "$ISCC" "/DMyAppVersion=$version" \
        "$WINREPO\\packaging\\installer.iss" 2>/dev/null | tail -2
    ls -lh dist/*setup.exe
}

run() {
    # Mesa beside the exe + the override is the Wine-only GL fix; the
    # installer itself stays clean of it
    for dll in $MESA_DLLS; do
        cp "$WINEPREFIX/drive_c/mesa/$dll" dist/VisualDynamics/
    done
    WINEDLLOVERRIDES='opengl32=n,b' GALLIUM_DRIVER=llvmpipe \
        "$WINE" 'Z:'"${REPO//\//\\}"'\dist\VisualDynamics\VisualDynamics.exe' "$@"
}

case "${1:-build}" in
    setup) setup ;;
    build) build ;;
    run)   shift; run "$@" ;;
    *) echo "usage: $0 setup|build|run" >&2; exit 2 ;;
esac
