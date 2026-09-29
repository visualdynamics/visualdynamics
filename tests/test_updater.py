"""The packaged macOS app updates itself through Sparkle (2026-09-28).

What can be pinned without a packaged app: the Info.plist the spec
writes (checks only when asked, the feed for the build's own
architecture, the key updates are verified against), the build script
putting Sparkle in without its sandbox-only services, and the window
handing Check for Updates to Sparkle where there is one and keeping
the manifest check where there is not. The update itself — download,
delta, signature, replacement — is Sparkle's, and is exercised on a
built app (PLAN.md, "The macOS app updates itself").
"""

from __future__ import annotations

import base64
import pathlib
import platform
import re

from visualdynamics import update
from visualdynamics.gui import updater

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = (ROOT / 'packaging' / 'visualdynamics.spec').read_text(encoding='utf-8')
BUILD = (ROOT / 'packaging' / 'build_macos.sh').read_text(encoding='utf-8')


def test_checks_happen_only_when_asked():
    """Brandon, 2026-09-28: keep checks manual. Sparkle checks on its
    own unless the Info.plist says not to, and on the second launch asks
    whether it may — both would contradict the privacy policy, which
    says the program goes online only when asked."""
    for key in ('SUEnableAutomaticChecks', 'SUAllowsAutomaticUpdates',
                'SUAutomaticallyUpdate'):
        assert re.search(rf"'{key}': False", SPEC), key


def test_each_architecture_reads_its_own_feed():
    """The arm64 and Intel builds are separate apps; an update for one
    must never reach the other."""
    assert "SPARKLE_FEED.format(arch=platform.machine())" in SPEC
    feed = update.SPARKLE_FEED.format(arch=platform.machine())
    assert feed.startswith('https://github.com/visualdynamics/visualdynamics/')
    assert feed.endswith(f'/latest/download/appcast-{platform.machine()}.xml')


def test_updates_are_verified_against_an_eddsa_key():
    """Sparkle's public key is 32 bytes of Ed25519, base64."""
    assert "'SUPublicEDKey': SPARKLE_PUBLIC_KEY" in SPEC
    assert len(base64.b64decode(update.SPARKLE_PUBLIC_KEY, validate=True)) == 32


def test_the_build_carries_sparkle_without_its_sandbox_services():
    """Sparkle needs its XPC services only in a sandboxed app; left in,
    they are two more nested bundles to sign for nothing."""
    assert 'Contents/Frameworks/Sparkle.framework' in BUILD
    assert 'XPCServices' in BUILD and 'rm -rf' in BUILD
    # and the archive Sparkle downloads is made from the stapled app
    staple = BUILD.index('xcrun stapler staple -q "$APP"')
    archive = BUILD.index('-macos-${ARCH}.zip')
    assert staple < archive


def test_outside_the_packaged_app_there_is_no_sparkle():
    """A pip install or a checkout has no framework to drive."""
    assert updater.framework_path() is None
    assert updater.controller() is None
    assert updater.check() is False


def test_the_menu_hands_the_check_to_sparkle(window, pump, monkeypatch):
    """Where Sparkle is, it does the whole job in its own window, and
    the manifest is not asked; where it is not, the manifest is."""
    asked = []
    monkeypatch.setattr(update, 'attempt',
                        lambda *a, **k: asked.append(1) or (None, 'offline'))
    monkeypatch.setattr(updater, 'check', lambda: True)
    window.check_for_updates()
    pump(5)
    assert asked == [], 'Sparkle checks; the manifest is not asked too'
    monkeypatch.setattr(updater, 'check', lambda: False)
    window.check_for_updates()
    for _ in range(50):
        pump(2)
        if asked:
            break
    assert asked == [1], 'without Sparkle, the manifest check'


def test_no_static_library_ships():
    """A static library is for linking, never loaded; the one PySide6's
    qml tree carries was signed as Mach-O, codesign kept that signature
    in extended attributes, and Sparkle refused to make a delta across
    it (2026-09-28). The spec leaves them all out."""
    assert "if not entry[0].endswith('.a')]" in SPEC
    assert SPEC.count("if not entry[0].endswith('.a')]") == 2, (
        'both the data and the binaries')


def test_the_release_script_survives_a_first_release():
    """A first release has no old deltas to clear, and zsh stops on a
    glob that matches nothing — the dry run of 2026-09-28 found it."""
    script = (ROOT / 'packaging' / 'release_updates.sh').read_text(
        encoding='utf-8')
    for line in script.splitlines():
        if line.strip().startswith('rm -f') and '*' in line:
            assert '(N)' in line, line
