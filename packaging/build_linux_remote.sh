#!/bin/zsh
# The Linux AppImage, built on the x86_64 Debian box (the media
# server) — this Mac has no container runtime, and PyInstaller only
# packages the OS it runs on.
#
#     packaging/build_linux_remote.sh
#
# Syncs the working tree over, runs the container recipe from
# packaging/README.md there, and brings the AppImage back into dist/.
# LAN first, Tailscale fallback, so it works from anywhere the server
# does. Serial and self-contained; refresh_builds.sh runs it in the
# background beside the local builds.
set -e
cd "$(dirname "$0")/.."
KEY="$HOME/.ssh/plex_migration_ed25519"
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=10)

HOST=192.168.10.107
ssh $SSH_OPTS -o ConnectTimeout=5 bzwink@$HOST true 2>/dev/null \
    || HOST=100.125.116.51
ssh $SSH_OPTS bzwink@$HOST true 2>/dev/null || {
    echo 'the build server is unreachable — no Linux build this round' >&2
    exit 1
}
echo "building the AppImage on $HOST"

rsync -a --delete -e "ssh ${SSH_OPTS[*]}" \
    --exclude .git --exclude .venv --exclude stressdata --exclude dist \
    --exclude build --exclude site --exclude __pycache__ \
    --exclude 'web/launch/downloads' --exclude '*.pyc' \
    ./ bzwink@$HOST:vd-build/

# the container recipe from packaging/README.md, verbatim in spirit:
# the apt list is the committed single source, appimagetool needs the
# extract-and-run switch anywhere FUSE is missing
# the container runs as root, so its dist/ and build/ come back
# root-owned and a plain rm is refused next round — sudo clears them
ssh $SSH_OPTS bzwink@$HOST 'cd vd-build && sudo rm -rf dist build && \
  docker run --rm --cpus 3 --memory 8g -v "$PWD":/work -w /work \
    -e APPIMAGE_EXTRACT_AND_RUN=1 python:3.14-slim-bookworm bash -c '\''
      set -e
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq
      grep -vE "^[[:space:]]*(#|$)" packaging/linux/apt-packages.txt | tr "\n" "\0" |
        xargs -0 apt-get satisfy -y -qq --no-install-recommends
      wget -q https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
      chmod +x appimagetool-x86_64.AppImage
      mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool
      pip install -q ".[step]" pyinstaller pillow
      PYTHON=python packaging/build_linux.sh'\'''

mkdir -p dist
scp $SSH_OPTS "bzwink@$HOST:vd-build/dist/*.AppImage" dist/
ls -lh dist/*.AppImage
