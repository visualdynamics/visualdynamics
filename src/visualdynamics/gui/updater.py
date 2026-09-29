"""The packaged macOS app updates itself, through Sparkle (2026-09-28).

Brandon: *is there a way to make the app update itself instead of making
me re-download the whole dmg every time?* Sparkle is the updater most
Mac applications outside the App Store use (MIT). It reads the feed
named in the bundle's Info.plist (`update.SPARKLE_FEED`), shows what is
new, downloads the smallest thing that gets there — a binary delta from
the installed version when the release carries one, the whole archive
otherwise — checks its EdDSA signature against the key in the
Info.plist and the new app's Developer ID against this one's, then
replaces the app and relaunches it.

**Only when asked.** The Info.plist sets `SUEnableAutomaticChecks` off,
so Sparkle never checks on its own and never asks to; *Check for
Updates* is still the one network request the program makes, as the
privacy policy says (Brandon, 2026-09-28: keep checks manual).

The framework is reached through pyobjc's runtime alone — no Cocoa
wrapper package — and only in the frozen macOS app, the one place
Sparkle.framework exists (`packaging/build_macos.sh` puts it in
`Contents/Frameworks`). Everywhere else `controller()` is None and the
window keeps the manifest check (`update.py`).
"""

from __future__ import annotations

import os
import sys
from typing import Any

#: the one controller, held for the life of the process: Sparkle's
#: updater is torn down with its last reference, mid-download included
_CONTROLLER: Any = None
_TRIED = False


def framework_path() -> str | None:
    """Sparkle.framework inside this app bundle, or None where there is
    none (not frozen, not macOS, or a build made without it)."""
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

    Started, because a controller that is not started refuses every
    check; with automatic checks off in the Info.plist, starting it
    schedules nothing. A framework that will not load answers None
    rather than raising: the menu item then falls back to the manifest
    check, which is worse than an update but better than an error.
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
    """Ask Sparkle to check now, with its own window; False when there
    is no Sparkle to ask."""
    updater = controller()
    if updater is None:
        return False
    updater.checkForUpdates_(None)
    return True
