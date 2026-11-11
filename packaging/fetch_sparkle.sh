#!/bin/bash
# The newest Sparkle release, unpacked into DEST, for a runner that has
# none (the release workflow's macos and updates jobs).
#
#     packaging/fetch_sparkle.sh DEST
#
# DEST/Sparkle.framework is what build_macos.sh copies into the app
# (SPARKLE), DEST/bin holds generate_appcast for release_updates.sh
# (SPARKLE_BIN). The newest release rather than a named one, as every
# dependency here tracks the latest (PRINCIPLES.md, 12); the archive is
# checked against the sha256 GitHub records for the asset before
# anything in it is unpacked, the same check the desk's copy had by
# hand (packaging/README.md, "Updating an installed copy").
set -euo pipefail
dest=${1:?where to unpack Sparkle}
read -r name url digest < <(
    gh api repos/sparkle-project/Sparkle/releases/latest --jq \
        '.assets[] | select(.name | test("^Sparkle-[0-9.]+\\.tar\\.xz$"))
         | "\(.name) \(.browser_download_url) \(.digest)"')
[[ $digest == sha256:* ]] || { echo "no digest recorded for $name" >&2; exit 1; }
mkdir -p "$dest"
# --fail so an error page is not saved as the archive, and a fresh
# download when the sum is wrong: 0.1.0a91's Intel updates job got a
# bad copy twice (2026-11-08) of the archive that summed right on the
# arm64 job beside it and on the desk
for try in 1 2 3; do
    curl -sSL --fail --retry 3 --retry-all-errors -o "$dest/$name" "$url" &&
        echo "${digest#sha256:}  $dest/$name" | shasum -a 256 -c - && break
    rm -f "$dest/$name"
    (( try < 3 )) || { echo "$name did not download whole in three tries" >&2; exit 1; }
    echo "$name: downloading again" >&2
done
tar -xJf "$dest/$name" -C "$dest"
rm "$dest/$name"
version=${name#Sparkle-}
echo "Sparkle ${version%.tar.xz} in $dest"
