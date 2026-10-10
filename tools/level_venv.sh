#!/bin/zsh
# Bring this desk's environments level with what CI installs, then gate.
#
#     tools/level_venv.sh            # upgrade both venvs, then the gate
#     tools/level_venv.sh --no-gate  # upgrade only
#
# The dependencies are unpinned (Principle 12), so every runner and
# every `pip install visualdynamics` gets the newest release of each —
# and the venvs here get nothing unless asked. They drifted: 0.1.0a41
# was cut from PySide6 6.11 and ruff 0.16 while the public runner had
# 6.12 and 0.17, and three syncs found what this would have found first
# (2026-10-09). The macOS images are built from these venvs, so a stale
# one ships older packages than CI tested. Step one of a release.
#
# `--upgrade-strategy eager` is the point: a plain `pip install -U`
# upgrades the package and leaves every dependency it already has.

set -u
cd "$(dirname "$0")/.." || exit 1
gate=1
[[ ${1:-} == --no-gate ]] && gate=0
scratch=$(mktemp -d)

level() {   # venv, then what it installs
  local venv=$1; shift
  [[ -x $venv/bin/python ]] || { echo "$venv: absent, skipped"; return 0; }
  "$venv/bin/python" -m pip freeze --exclude-editable > "$scratch/before" 2>/dev/null
  "$venv/bin/python" -m pip install -q -U --upgrade-strategy eager \
      --prefer-binary "$@" || { echo "$venv: pip failed" >&2; return 1; }
  "$venv/bin/python" -m pip freeze --exclude-editable > "$scratch/after" 2>/dev/null
  echo "$venv ($("$venv/bin/python" -c 'import platform; print(platform.python_version(), platform.machine())')):"
  # name==old -> new for each package that moved, and what is new
  python3 - "$scratch/before" "$scratch/after" <<'PY'
import sys
def read(path):
    pins = {}
    for line in open(path):
        name, sep, version = line.strip().partition('==')
        if sep:
            pins[name.lower()] = version
    return pins
before, after = read(sys.argv[1]), read(sys.argv[2])
moved = [f'  {n} {before[n]} -> {v}' for n, v in sorted(after.items())
         if n in before and before[n] != v]
added = [f'  {n} {v} (new)' for n, v in sorted(after.items()) if n not in before]
print('\n'.join(moved + added) if moved or added else '  already level')
PY
}

level .venv -e '.[dev,step,docs,app]' pyinstaller pillow || exit 1
level .venv-x86_64 '.[step,app]' pyinstaller pillow || exit 1

# the two builds ship together, so the interpreter and the libraries
# they freeze must agree — one interpreter start per venv, the Intel
# one under Rosetta being the slow one
frozen='import importlib.metadata as m, platform
print("CPython", platform.python_version())
for name in ("PySide6", "numpy", "scipy", "vtk", "pyvista"):
    print(name, m.version(name))'
if [[ -x .venv-x86_64/bin/python ]]; then
  diff <(.venv/bin/python -c "$frozen") <(.venv-x86_64/bin/python -c "$frozen") \
      >/dev/null || {
    echo "warning: .venv and .venv-x86_64 differ:" >&2
    diff <(.venv/bin/python -c "$frozen") <(.venv-x86_64/bin/python -c "$frozen") \
        | grep '^[<>]' >&2
  }
fi

(( gate )) || exit 0
./.venv/bin/ruff check . || exit 1
# judged by pytest's own exit code, its output in a file — never a pipe
./.venv/bin/python -m pytest tests -q -n 4 -o faulthandler_timeout=300 \
    > "$scratch/gate.txt" 2>&1
code=$?
tail -1 "$scratch/gate.txt"
grep -E '^(FAILED|ERROR)' "$scratch/gate.txt"
echo "gate exit $code (full output: $scratch/gate.txt)"
exit $code
