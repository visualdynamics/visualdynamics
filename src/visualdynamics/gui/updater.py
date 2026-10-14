"""Updating the packaged macOS app in place, through Sparkle.

Sparkle (MIT licensed) is the updater most Mac applications outside the
App Store use. When asked to check, it reads the feed named by
`SUFeedURL` in the app's Info.plist (`update.SPARKLE_FEED`, one feed per
architecture, since the arm64 and Intel builds are separate apps) and
shows what is new. It then downloads a binary delta from the installed
version when the release has one, or the whole archive otherwise,
checks the download's EdDSA signature against `SUPublicEDKey` in the
Info.plist and the new app's Developer ID signature against the
installed app's, replaces the app and relaunches it.

It checks only when asked. The Info.plist turns off
`SUEnableAutomaticChecks`, `SUAllowsAutomaticUpdates` and
`SUAutomaticallyUpdate`, so Sparkle never checks on its own, never asks
for permission to, and never downloads in the background. File → Check
for Updates is the only time the program makes a network request, as
the privacy policy says.

The framework is loaded through pyobjc's runtime (`pyobjc-core`), with
no Cocoa wrapper package, and only in the frozen macOS app, the one
place Sparkle.framework exists (`packaging/build_macos.sh` copies it
into `Contents/Frameworks`). Everywhere else `controller()` returns
None and Check for Updates uses the manifest check in `update.py`.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Brandon, 2026-09-28: "is there a way to make the app update itself
# instead of making me re-download the whole dmg every time?" Sparkle
# was the answer, and the same day he decided the checks stay manual
# (the Info.plist settings in packaging/visualdynamics.spec).

#: the one controller, held for the life of the process: Sparkle's
#: updater is torn down with its last reference, mid-download included
_CONTROLLER: Any = None
_TRIED = False


def framework_path() -> str | None:
    """The path to Sparkle.framework inside this app bundle, or None
    where there is none: not frozen, not macOS, no pyobjc, or a build
    made without the framework."""
    if sys.platform != 'darwin' or not getattr(sys, 'frozen', False):
        return None
    try:
        import objc
    except ImportError:
        return None
    bundle = objc.lookUpClass('NSBundle').mainBundle()
    frameworks = bundle.privateFrameworksPath()
    if not frameworks:
        return None
    path = os.path.join(str(frameworks), 'Sparkle.framework')
    return path if os.path.isdir(path) else None


def controller() -> Any:
    """Sparkle's standard updater controller, started once, or None.

    The first call decides and later calls return the same answer. The
    controller is started because one that is not started refuses every
    check; with automatic checks off in the Info.plist, starting it
    schedules nothing. If the framework does not load, this returns None
    instead of raising, and the menu item falls back to the manifest
    check, which can only report that a newer version exists.
    """
    global _CONTROLLER, _TRIED
    if _TRIED:
        return _CONTROLLER
    _TRIED = True
    path = framework_path()
    if path is None:
        return None
    try:
        import objc

        found: dict[str, Any] = {}
        objc.loadBundle('Sparkle', found, bundle_path=path)
        standard = objc.lookUpClass('SPUStandardUpdaterController')
        _CONTROLLER = (standard.alloc()
                       .initWithStartingUpdater_updaterDelegate_userDriverDelegate_(
                           True, None, None))
    except Exception:   # noqa: BLE001 — any failure means "no updater",
        _CONTROLLER = None                  # and the manifest check
    return _CONTROLLER


def check() -> bool:
    """Ask Sparkle to check now, in its own window. Returns True when it
    was asked, and False when there is no Sparkle to ask (see
    `controller`), so the caller can use the manifest check instead."""
    updater = controller()
    if updater is None:
        return False
    updater.checkForUpdates_(None)
    return True
