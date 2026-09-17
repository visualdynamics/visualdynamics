#!/usr/bin/env python3
"""Stage packaged builds for the launch site's local preview.

    packaging/stage_downloads.py          # stage what dist/ holds
    packaging/stage_downloads.py clear    # take it all back out

Hardlinks the installers the platform build scripts left in ``dist/``
into ``web/launch/downloads/`` and writes ``manifest.json`` beside
them. The downloads page's own script reads that manifest when the
site is served locally, so the preview's four download slots hand out
the installers this machine just built — the same click a visitor
makes against a GitHub release once one exists.

The staging directory is gitignored and must never travel:
``wrangler pages deploy`` uploads a directory as it stands, gitignore
notwithstanding, so run ``clear`` before any deploy of ``web/launch``.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STAGE = os.path.join(ROOT, 'web', 'launch', 'downloads')

#: what a slot on the downloads page can hold, by filename
KINDS = ('.dmg', '-setup.exe', '.AppImage')


def version_text(name: str) -> str:
    """'0.1.0a1' from 'VisualDynamics-0.1.0a1-macos-arm64.dmg'; '' for a
    name that carries no version."""
    parts = name.split('-')
    return parts[1] if len(parts) > 1 and parts[1][:1].isdigit() else ''


def version_of(name: str) -> tuple[int, ...]:
    """The name's version as the update check orders them — an alpha's
    files below its release's (2026-09-11: '0.1.0a1'); () for a name
    that carries no version, which sorts below every real one.

    Its own small parser rather than `visualdynamics.update`'s: this
    script runs under whatever `python3` the shell finds — the refresh
    script calls it by its shebang — and importing the package from it
    took the whole refresh down after the first build (2026-09-11).
    The rule is the same one: a, b, rc below the release, in that order.
    """
    text = version_text(name)
    if not text:
        return ()
    release, stage, number = text, 3, 0
    for rank, tag in enumerate(('a', 'b', 'rc')):
        head, found, tail = text.partition(tag)
        if found and head and head[-1].isdigit() and (tail == '' or tail.isdigit()):
            release, stage, number = head, rank, int(tail or 0)
            break
    parts = [int(''.join(c for c in piece if c.isdigit()) or 0)
             for piece in release.split('.')]
    while len(parts) < 3:
        parts.append(0)
    return (*parts, stage, number)


def manifest_for(entries: list[tuple[str, int]]) -> dict:
    """The manifest, from [(filename, bytes)] — pure, so testable.

    The shape mirrors what the page already reads from the GitHub
    releases API — one release: ``assets`` with ``name``/``size``/URL
    — so the page's one filling routine serves both sources. One
    *release*: staging is cumulative and older installers stay on
    disk, but only the newest version's files are listed, because the
    page takes the first file matching a platform and a name-sorted
    list put 0.0.1 ahead of 0.1.0 — Brandon downloaded the old,
    unsigned image from the preview the evening the notarized one
    landed (2026-09-02).
    """
    newest = max((version_of(name) for name, _size in entries),
                 default=())
    listed = sorted((name, size) for name, size in entries
                    if version_of(name) == newest)
    return {'local': True,
            # the tag is the version as written on the file, not
            # rebuilt from the sort key — which would say v0.1.0.3.0
            'tag_name': 'v' + version_text(listed[0][0]) if listed else '',
            'assets': [{'name': name, 'size': size,
                        'browser_download_url': f'downloads/{name}'}
                       for name, size in listed]}


def stage() -> None:
    """Cumulative on purpose: each platform's build script wipes
    dist/ before building, so at most one platform's installer is
    there at a time — the staged hardlinks survive the wipe, and the
    manifest is written from what is *staged*, not from what dist
    happens to hold this minute. `clear` is the reset."""
    os.makedirs(STAGE, exist_ok=True)
    dist = os.path.join(ROOT, 'dist')
    for name in sorted(os.listdir(dist)) if os.path.isdir(dist) else []:
        if not any(name.endswith(kind) for kind in KINDS):
            continue
        target = os.path.join(STAGE, name)
        if os.path.exists(target):
            os.unlink(target)
        os.link(os.path.join(dist, name), target)
        print('staged', name)
    entries = [(name, os.path.getsize(os.path.join(STAGE, name)))
               for name in sorted(os.listdir(STAGE))
               if any(name.endswith(kind) for kind in KINDS)]
    with open(os.path.join(STAGE, 'manifest.json'), 'w',
              encoding='utf-8') as out:
        json.dump(manifest_for(entries), out, indent=1)
    if not entries:
        print('nothing staged and nothing in dist/ to stage — '
              'build first; manifest written empty')


def clear() -> None:
    if os.path.isdir(STAGE):
        for name in os.listdir(STAGE):
            os.unlink(os.path.join(STAGE, name))
        os.rmdir(STAGE)
    print('staging cleared — web/launch is deployable again')


if __name__ == '__main__':
    {'clear': clear}.get(sys.argv[1] if len(sys.argv) > 1 else '', stage)()
