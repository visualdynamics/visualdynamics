#!/bin/zsh
# Sync the public repository the only way its main accepts a change:
# on a branch, through a pull request, merged by GitHub once CI is
# green (Brandon, 2026-09-14: a hard rule, administrators included).
#
#     tools/sync_public.sh "<the pull request's title>"
#     tools/sync_public.sh --release "<what the release brings>"
#
# --release asks for a release as well (Brandon, 2026-10-21): it sets
# __version__ to the next alpha (tools/next_alpha.py), commits that and
# pushes this repository's main, then syncs. Once the tests pass on the
# public main, the release workflow sees a version with no release yet
# and releases it, published with no one to wait for (Brandon,
# 2026-10-22; packaging/release_decision.py). It refuses
# uncommitted work: a release is what is committed, and nothing else.
#
# Builds the public tree into the clone, commits it on a branch named
# for the moment, pushes the branch, opens the pull request and marks
# it to merge itself when the checks pass — then waits, and reports.
# A red run leaves the pull request open with the failure on it; fix
# here, run this again, and the new sync lands on the same branch.
#
# One-time settings on the repository this leans on: auto-merge
# enabled (`gh repo edit --enable-auto-merge` — the first run stopped
# at "Auto merge is not allowed", 2026-09-15) and rebase merges
# allowed, so the sync commit lands as itself on a linear main.
set -e
cd "$(dirname "$0")/.."
CLONE="$HOME/visualdynamics-shared"
REPO=visualdynamics/visualdynamics
RELEASE=0
if [[ ${1:-} == --release ]]; then
    RELEASE=1
    shift
fi
TITLE=${1:?a title for the pull request}
if (( RELEASE )); then
    if [[ $(git rev-parse --abbrev-ref HEAD) != main ]]; then
        echo 'a release is cut from main'; exit 1
    fi
    if [[ -n "$(git status --porcelain)" ]]; then
        echo 'commit first: a release is built from what is committed'; exit 1
    fi
    version=$(./.venv/bin/python tools/next_alpha.py)
    git commit -q -m "$version" -- src/visualdynamics/__init__.py
    git push -q origin main
    TITLE="$version: $TITLE"
    echo "releasing $version: it publishes itself once the tests pass on main"
fi
IDENT=(-c user.email=269075368+bzwink@users.noreply.github.com -c user.name="Brandon Zwink")

# Settle the branch before building into it. The clone is left on a
# sync branch when a pull request's checks failed, so that the next
# run commits the fix onto the same branch and the same pull request
# — but a branch whose pull request is *closed* (or merged, or gone)
# is spent: committing onto it re-creates a deleted branch and the
# auto-merge call answers "pull request is closed" (2026-09-16, after
# #11 was closed by hand). Only an open pull request keeps its branch.
cd "$CLONE"
git fetch -q origin main
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ $branch != main ]]; then
    state=$(gh pr view "$branch" --repo "$REPO" --json state -q .state 2>/dev/null || echo NONE)
    if [[ $state != OPEN ]]; then
        git checkout -q -f -B main origin/main
        branch=main
    fi
fi
if [[ $branch == main ]]; then
    git checkout -q -f -B main origin/main
fi
cd - > /dev/null
./.venv/bin/python tools/build_public_tree.py "$CLONE" --sync > /dev/null
cd "$CLONE"
if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
    echo 'nothing to sync: the public tree already matches'; exit 0
fi
if [[ $branch == main ]]; then
    branch="sync/$(date -u +%Y%m%d-%H%M)"
    git checkout -q -b "$branch"
fi
git add -A
git "${IDENT[@]}" commit -q -m "$TITLE"
git push -q -u origin "$branch"
if ! gh pr view "$branch" --repo "$REPO" > /dev/null 2>&1; then
    gh pr create --repo "$REPO" --base main --head "$branch" \
        --title "$TITLE" --body "Synced from the working tree by tools/sync_public.sh; merges itself once the checks are green." > /dev/null
fi
gh pr merge "$branch" --repo "$REPO" --auto --rebase --delete-branch > /dev/null
url=$(gh pr view "$branch" --repo "$REPO" --json url -q .url)
echo "pull request $url — waiting on the checks"
# the checks take a moment to be reported after the pull request opens,
# and --watch on a pull request with none yet answers "no checks
# reported" and fails (the second gated sync, 2026-09-15)
for i in $(seq 1 30); do
    gh pr checks "$branch" --repo "$REPO" > /dev/null 2>&1 && break
    sleep 10
done
gh pr checks "$branch" --repo "$REPO" --watch --fail-fast > /dev/null || {
    echo "a check failed — the pull request stays open: $url" >&2; exit 1; }
# GitHub merges on its own once the checks are green; wait for it
for i in $(seq 1 60); do
    state=$(gh pr view "$branch" --repo "$REPO" --json state -q .state)
    [[ $state == MERGED ]] && break
    sleep 5
done
[[ $state == MERGED ]] || { echo "green, but not merged yet: $url" >&2; exit 1; }
git checkout -q main && git fetch -q origin && git reset -q --hard origin/main
git branch -q -D "$branch" 2>/dev/null || true
echo "merged into main: $(git log --oneline -1)"
