#!/bin/zsh
# The Intel dmg, cross-built on this Apple-silicon Mac under Rosetta.
#
#     packaging/build_macos_intel.sh
#
# The mirror of the Wine route, with less friction: Rosetta runs
# x86_64 processes natively, so an x86_64 Python (unpacked from
# python-build-standalone — no installer, no sudo) pulls x86_64
# wheels and PyInstaller emits an x86_64 app, which this machine can
# even launch to test. A single universal2 bundle is NOT buildable —
# VTK ships single-architecture wheels — so macOS is two dmgs, the
# way most Qt applications ship it (decided 2026-08-31, reversing the
# same day's Apple-silicon-only call).
set -e
cd "$(dirname "$0")/.."

BASE="$HOME/.python-x86_64"
VENV=".venv-x86_64"
PBS=https://github.com/astral-sh/python-build-standalone/releases/download/20260825/cpython-3.13.15%2B20260825-x86_64-apple-darwin-install_only.tar.gz

if [[ ! -x "$BASE/python/bin/python3.13" ]]; then
    echo '== x86_64 CPython (python-build-standalone; no installer)'
    work=$(mktemp -d)
    curl -sL -o "$work/python.tar.gz" "$PBS"
    mkdir -p "$BASE"
    tar -xzf "$work/python.tar.gz" -C "$BASE"
    rm -rf "$work"
fi
"$BASE/python/bin/python3.13" -c \
    'import platform; assert platform.machine() == "x86_64", platform.machine()'

if [[ ! -x "$VENV/bin/python" ]]; then
    echo '== the x86_64 venv'
    "$BASE/python/bin/python3.13" -m venv "$VENV"
fi
echo '== the project and its build tools, as x86_64 wheels'
"$VENV/bin/python" -m pip install -q --prefer-binary '.[step]' pyinstaller pillow

# The x86_64 PyInstaller thins its bootloader with lipo, and the
# signing and notarizing that follow call codesign, notarytool and
# stapler — all through xcrun. The Command Line Tools' xcrun library
# is arm64-only, so under Rosetta it cannot even load (the machine's
# selection fell back to the CLT when Xcode became Xcode-beta,
# 2026-09-02, and the Intel build died at lipo). An Xcode's toolchain
# is universal: name it for this build alone, without touching the
# machine's xcode-select, preferring a release Xcode to a beta.
if [[ -z ${DEVELOPER_DIR:-} && "$(xcode-select -p)" == *CommandLineTools* ]]; then
    for xcode in /Applications/Xcode.app /Applications/Xcode-beta.app; do
        if [[ -d "$xcode/Contents/Developer" ]]; then
            export DEVELOPER_DIR="$xcode/Contents/Developer"
            echo "== toolchain: $xcode (the Command Line Tools cannot run under Rosetta)"
            break
        fi
    done
fi

PYTHON="./$VENV/bin/python" ./packaging/build_macos.sh
