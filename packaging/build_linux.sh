#!/usr/bin/env bash
# Visual Dynamics as an AppImage: one file, chmod +x, run.
#
#     packaging/build_linux.sh
#
# Needs appimagetool on PATH. Without it the directory under dist/ is a
# working application; the AppImage is the part that makes it portable
# across distributions.
set -euo pipefail
cd "$(dirname "$0")/.."

# the repo venv when there is one; a runner or a container has none and
# installed into its own interpreter (the release workflow's first Linux
# run stopped here, 2026-09-01, on a venv that does not exist there)
PYTHON=${PYTHON:-$([[ -x ./.venv/bin/python ]] && echo ./.venv/bin/python || echo python3)}
VERSION=$("$PYTHON" -c "import visualdynamics; print(visualdynamics.__version__)")

# With INSTALL_DEPS=1 (a container or a fresh runner), install what the
# build needs from the one list. Left off by default: a developer's own
# machine already has these and should not be apt-getting behind their
# back.
if [[ ${INSTALL_DEPS:-0} == 1 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  grep -vE '^\s*(#|$)' packaging/linux/apt-packages.txt | tr '\n' '\0' |
    xargs -0 apt-get satisfy -y -qq --no-install-recommends >/dev/null
fi

rm -rf build dist
"$PYTHON" -m PyInstaller packaging/visualdynamics.spec --noconfirm \
  --distpath dist --workpath build

# the bundle loads oneMKL and factors with PARDISO (core.sparse_solve.check):
# a build that left it out still runs, through SuperLU, many times slower
# on a large model, and only this says so. Before Qt, so no display needed
dist/VisualDynamics/VisualDynamics --check-solver

if ! command -v appimagetool >/dev/null; then
  echo "appimagetool not found; dist/VisualDynamics/ is the application." >&2
  exit 0
fi
ROOT=dist/AppDir
rm -rf "$ROOT"; mkdir -p "$ROOT/usr/bin"
cp -r dist/VisualDynamics/. "$ROOT/usr/bin/"
cp packaging/linux/visualdynamics.desktop "$ROOT/"
cp packaging/icon-512.png "$ROOT/visualdynamics.png"
cp packaging/linux/AppRun "$ROOT/AppRun"; chmod +x "$ROOT/AppRun"
appimagetool "$ROOT" "dist/VisualDynamics-${VERSION}-linux-x86_64.AppImage"
echo "built dist/VisualDynamics-${VERSION}-linux-x86_64.AppImage"
