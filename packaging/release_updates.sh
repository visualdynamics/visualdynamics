#!/bin/zsh
# Release day, after attach_macos.sh: the macOS app's Sparkle updates
# (src/visualdynamics/gui/updater.py, 2026-09-28).
#
#     packaging/release_updates.sh v0.1.0a15
#
# For each architecture — the arm64 and Intel builds are separate apps
# with separate feeds — it puts this release's update archive (the zip
# build_macos.sh writes beside the dmg) in the archive store, has
# Sparkle's generate_appcast sign it and make binary deltas from the
# two releases before it, and uploads the archive, the deltas and
# appcast-<arch>.xml to the draft release. The feed every app reads is
# `releases/latest/download/appcast-<arch>.xml`, so publishing the
# release is what publishes the update.
#
# The store (UPDATE_STORE) is kept on this machine, not in the
# repository: the deltas are computed against the previous releases'
# own archives, so it holds the last three of each. The signing key is
# the login-keychain item generate_keys made (account `visualdynamics`),
# read without a prompt while Brandon is logged in.
#
# VD_DRY_RUN=1 does everything but talk to GitHub — no SHA256SUMS
# fetched, nothing uploaded, the notes left out — which is how the
# delta naming was exercised on test archives before a real release.
set -e
cd "$(dirname "$0")/.."
tag=${1:?the release tag, e.g. v0.1.0a15}
repo=visualdynamics/visualdynamics
version=${tag#v}
release=${VD_RELEASE_DIR:-"$HOME/Library/Application Support/visualdynamics-release"}
tools=${SPARKLE_BIN:-"$release/sparkle/bin"}
store=${UPDATE_STORE:-"$release/updates"}
[[ -x $tools/generate_appcast ]] || { echo "no Sparkle tools at $tools" >&2; exit 1; }

work=$(mktemp -d)
notes=
if [[ -z ${VD_DRY_RUN:-} ]]; then
    gh release download "$tag" --repo "$repo" --pattern SHA256SUMS --dir "$work" \
        || { echo "the draft release $tag has no SHA256SUMS yet — attach the images first" >&2; exit 1; }
    # the release's own notes, as the text Sparkle shows beside Install
    notes=$(gh release view "$tag" --repo "$repo" --json body --jq .body |
            sed -n 's/^\* \(.*\) by @[^ ]* in .*/<li>\1<\/li>/p')
else
    touch "$work/SHA256SUMS"
fi

uploads=()
# VD_ARCHES narrows it to one architecture, for a test
for arch in ${=VD_ARCHES:-arm64 x86_64}; do
    archive=dist/VisualDynamics-${version}-macos-${arch}.zip
    [[ -f $archive ]] || { echo "no $archive — build_macos.sh writes it once the app is stapled" >&2; exit 1; }
    dir=$store/$arch
    mkdir -p "$dir"
    cp "$archive" "$dir/"
    [[ -n $notes ]] && print -r -- "<ul>$notes</ul>" > "$dir/${archive:t:r}.html"
    # the last three releases stay, this one included: deltas reach back
    # two, and anything older gets the whole archive
    ls -t "$dir"/VisualDynamics-*-macos-${arch}.zip | tail -n +4 |
        while read -r old; do rm -f "$old" "${old:r}.html"; done
    # (N): a first release has no old deltas, and zsh stops on a glob
    # that matches nothing
    rm -f "$dir"/*.delta(N) "$dir"/appcast-*.xml(N)
    "$tools/generate_appcast" --account visualdynamics \
        --download-url-prefix "https://github.com/$repo/releases/download/$tag/" \
        --embed-release-notes --link https://visualdynamics.org \
        --maximum-versions 1 --maximum-deltas 2 \
        -o "$dir/appcast-${arch}.xml" "$dir"
    # GitHub turns a space in an asset's name into a dot, and Sparkle
    # names deltas after the app ("Visual Dynamics"), so the names and
    # the feed's links are changed together; the signatures are of the
    # files' contents and do not care
    for delta in "$dir"/*.delta(N); do
        plain=${delta:t:gs/ /./}
        mv "$delta" "$dir/${plain:r}-${arch}.delta"
        sed -i '' "s|${${delta:t}// /%20}|${plain:r}-${arch}.delta|g" "$dir/appcast-${arch}.xml"
        uploads+=("$dir/${plain:r}-${arch}.delta")
    done
    uploads+=("$archive" "$dir/appcast-${arch}.xml")
    name=${archive:t}
    grep -q "  $name\$" "$work/SHA256SUMS" && sed -i '' "/  $name\$/d" "$work/SHA256SUMS"
    shasum -a 256 "$archive" | sed "s|  .*|  $name|" >> "$work/SHA256SUMS"
    echo "$arch: $(grep -c '<item>' "$dir/appcast-${arch}.xml") item, $(grep -c 'sparkle:deltaFrom' "$dir/appcast-${arch}.xml") deltas"
done
# generate_appcast unpacks every archive it diffs into a cache of its
# own, about 2 GB a release, and never empties it (5.9 GB after one
# test, 2026-09-28); the next run unpacks what it needs again
rm -rf ~/Library/Caches/Sparkle_generate_appcast
if [[ -n ${VD_DRY_RUN:-} ]]; then
    echo "dry run — would upload to $tag:"
else
    gh release upload "$tag" --repo "$repo" --clobber "${uploads[@]}" "$work/SHA256SUMS"
    echo "uploaded to $tag:"
fi
printf '  %s\n' "${uploads[@]:t}"
