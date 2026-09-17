#!/bin/zsh
# Release day, step 5: the two macOS images built and notarized on
# this Mac go onto the draft release the workflow opened, and their
# checksums join the SHA256SUMS the workflow wrote for the runner
# builds — so one file verifies every asset (2026-09-11).
#
#     packaging/attach_macos.sh v0.1.0a1
#
# Idempotent: uploads clobber, and a line already in SHA256SUMS is
# not written twice. Needs `gh` signed in to the shared repository's
# owner, and both images in dist/ for the tag's version.
set -e
cd "$(dirname "$0")/.."
tag=${1:?the release tag, e.g. v0.1.0a1}
repo=visualdynamics/visualdynamics
version=${tag#v}
images=(dist/VisualDynamics-${version}-macos-*.dmg)
[[ ${#images} -eq 2 ]] || { echo "expected two images for $version in dist/, found: $images" >&2; exit 1; }
for image in $images; do
    xcrun stapler validate "$image" > /dev/null || { echo "$image is not stapled" >&2; exit 1; }
done
work=$(mktemp -d)
gh release download "$tag" --repo "$repo" --pattern SHA256SUMS --dir "$work" \
    || { echo "the draft release $tag has no SHA256SUMS yet — has the workflow finished?" >&2; exit 1; }
for image in $images; do
    name=$(basename "$image")
    line=$(shasum -a 256 "$image" | sed "s|  .*|  $name|")
    grep -q "  $name\$" "$work/SHA256SUMS" && sed -i '' "/  $name\$/d" "$work/SHA256SUMS"
    echo "$line" >> "$work/SHA256SUMS"
done
gh release upload "$tag" --repo "$repo" --clobber $images "$work/SHA256SUMS"
echo "attached to $tag:"
cat "$work/SHA256SUMS"
