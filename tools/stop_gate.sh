#!/bin/sh
# Stop a stalled test gate, all of it.
#
# `pkill -f "pytest tests"` kills the pytest parent and nothing else:
# its xdist workers run as `python -u -c …` and survive, each parked in
# the teardown that stalled, holding a QtWebEngine render process. On
# 2026-10-02 three gates' worth of such workers were still alive two
# and a half hours on, and every gate run beside them stalled in its
# own web-page teardown. This stops the parent, every worker of this
# repository's venv that the parent left behind, their children, and
# the orphaned web engine helpers — never the helpers of an open app.
# the venv's python, however the gate was started: a relative
# `./.venv/bin/python` or the absolute path both end in this
venv='visualdynamics/.venv/bin/python'
case "$(pwd)" in */visualdynamics) ;; *) cd "$(dirname "$0")/.." || exit 1 ;; esac
pkill -9 -f '\.venv/bin/python -m pytest' 2>/dev/null
sleep 2
# the workers, now orphaned and reparented to launchd, or still
# waiting on a parent that is gone
for pid in $(ps -axo pid=,command= | awk '/\.venv\/bin\/python -u/ && !/awk/ {print $1}'); do
    pkill -9 -P "$pid" 2>/dev/null
    kill -9 "$pid" 2>/dev/null
done
# web engine helpers the workers left orphaned — only those whose
# parent is gone, never the helpers of an app that is open
for pid in $(ps -axo pid=,ppid=,command= | awk '$2 == 1 && /[Q]tWebEngineProcess/ {print $1}'); do
    kill -9 "$pid" 2>/dev/null
done
sleep 1
left=$(ps -axo command= | grep -c '[.]venv/bin/python -u' | tr -d ' ')
helpers=$(ps -axo ppid=,command= | awk '$1 == 1 && /[Q]tWebEngineProcess/' | wc -l | tr -d ' ')
echo "stopped: $left workers and $helpers orphaned web engine helpers left"
: "$venv"
