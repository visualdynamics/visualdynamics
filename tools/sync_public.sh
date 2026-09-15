#!/bin/zsh
# Sync the public repository the only way its main accepts a change:
# on a branch, through a pull request, merged by GitHub once CI is
# green (Brandon, 2026-09-14: a hard rule, administrators included).
#
#     tools/sync_public.sh "<the pull request's title>"
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
TITLE=${1:?a title for the pull request}
IDENT=(-c user.email=269075368+bzwink@users.noreply.github.com -c user.name="Brandon Zwink")

./.venv/bin/python tools/build_public_tree.py "$CLONE" --sync > /dev/null
cd "$CLONE"
if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
    echo 'nothing to sync: the public tree already matches'; exit 0
fi
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ $branch == main ]]; then
    git fetch -q origin main && git checkout -q -B main origin/main
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
