"""Check whether a newer version of Visual Dynamics is available.

This module checks; it does not update. It reads one small JSON file,
the manifest, from one URL, and the app reports the answer in its status
line (*0.1.0 is available*), with a dialog when a newer version exists.
Nothing is downloaded or replaced, and nothing runs with privileges.

The request is made only when someone asks for it: in the app, File >
Check for Updates, and never at startup; from a script, a call to
`attempt`, `fetch` or `check`. The only information it sends about
the program is a User-Agent header, `VisualDynamics/` followed by the
version number. Like any web request it reaches the host from the
machine's IP address, and the program keeps no record of the check.

This is all the updating there is except in the packaged macOS app. An
updater has to verify what it downloaded before running it, or anyone
who can change the download can run code on the machine. The macOS app
updates itself through Sparkle (`gui/updater.py`): every update is
signed with an EdDSA key held only by the release process, and the app
refuses anything that does not verify against `SPARKLE_PUBLIC_KEY`.
Windows will get WinSparkle once its installer is signed, and a pip
install updates with pip (`upgrade_command`).

The manifest is served from `visualdynamics.org`, a static host. The
release workflow writes it from each published release as
`web/launch/latest.json` and deploys it with the site. Its shape:

    {"version": "0.1.0", "url": "https://.../releases", "notes": "..."}
"""

from __future__ import annotations

import json
import ssl
from typing import Any
from urllib.request import Request, urlopen

from . import __version__

# Stopping at a check was a deliberate choice, and it still holds
# everywhere but the packaged macOS app, which has updated itself
# through Sparkle since 2026-09-28.

MANIFEST = 'https://visualdynamics.org/latest.json'
TIMEOUT = 4.0

#: the macOS app's Sparkle feed, one per architecture — the two builds
#: are separate apps, so each has its own list. A release asset rather
#: than a page on the site: GitHub answers `latest/download/<name>` with
#: whatever the newest published release carries, so the list and the
#: archives it names are published in the one step that publishes the
#: release (`packaging/release_updates.sh`)
SPARKLE_FEED = ('https://github.com/visualdynamics/visualdynamics/releases/'
                'latest/download/appcast-{arch}.xml')
#: the public half of the EdDSA key every Sparkle update is signed with;
#: the private half is a login-keychain item on the release machine,
#: under the account `visualdynamics` (`generate_keys --account
#: visualdynamics`), and a copy of it is the release workflow's
#: SPARKLE_ED_KEY secret
SPARKLE_PUBLIC_KEY = 'mm825xk4k0HFYxuJGzmsi/df6IEdrv9GtBmMP+KzB/A='


def parse_version(text: str) -> tuple[int, ...]:
    """A version string as a tuple to compare: '0.10.2' -> (0, 10, 2, 3, 0).

    The parts are numbers, so 0.10 sorts above 0.9; compared as strings
    they would sort the other way and offer a downgrade as an upgrade. A
    leading 'v' is ignored, and a part that cannot be read becomes zero
    instead of raising, so a malformed manifest means no update is
    offered.

    The last two entries are the stage and its number. A pre-release
    sorts below its release: '0.1.0a1' is (0, 1, 0, 0, 1) and '0.1.0' is
    (0, 1, 0, 3, 0). The stage is 0 for a (alpha), 1 for b (beta), 2 for
    rc (release candidate), spelled as Python spells them, and 3 for a
    release.

    Parameters
    ----------
    text : str
        The version, as the manifest or `__version__` spells it.

    Returns
    -------
    tuple of int
        Major, minor and patch (missing parts are zero), then the stage
        and its number.
    """
    # Pre-releases sort below their release so the first non-alpha
    # 0.1.0 is offered to an alpha as an upgrade (2026-09-11).
    text = str(text).strip().lstrip('v')
    release, stage, number = text, 3, 0
    for rank, tag in enumerate(('a', 'b', 'rc')):
        head, found, tail = text.partition(tag)
        if found and head and head[-1].isdigit() and (tail == '' or tail.isdigit()):
            release, stage, number = head, rank, int(tail or 0)
            break
    parts: list[int] = []
    for piece in release.split('.'):
        digits = ''.join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return (*parts, stage, number)


def newer(available: str, running: str = __version__) -> bool:
    """Whether `available` is a later version than `running`, which
    defaults to the version running now. Compared by `parse_version`."""
    return parse_version(available) > parse_version(running)


def trust_store() -> ssl.SSLContext:
    """The TLS context the check verifies visualdynamics.org with: the
    certificates the operating system trusts.

    With `truststore` installed (MIT licensed, and what pip uses), the
    context verifies against the system's own store: the Keychain on
    macOS, the certificate store on Windows, OpenSSL's system bundle on
    Linux. A network that inspects HTTPS re-signs every site with its
    own certificate authority, which IT installs in that store, so the
    check works on such a network wherever the browser and `curl` do.

    Without `truststore`, the context is Python's default one. When that
    holds no certificates, the bundle `certifi` ships is loaded into it,
    and when `certifi` is not installed either, the fetch fails with a
    certificate error.
    """
    # Brandon's work machine reported "unable to get local issuer
    # certificate" on a network where `curl` got a 200 (2026-09-25): a
    # Python bundle of Mozilla's roots does not hold the network's own
    # authority, which is why the system store comes first.
    #
    # The default context can be empty because a packaged build carries
    # its own OpenSSL, and the one the wheels bring was built elsewhere
    # with a compiled-in certificate directory that exists on nobody
    # else's machine (read off the library with `strings`, 2026-09-15).
    try:
        import truststore
    except ImportError:
        pass
    else:
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context = ssl.create_default_context()
    if not context.get_ca_certs():
        try:
            import certifi
            context.load_verify_locations(certifi.where())
        except Exception:  # noqa: BLE001, S110 — no bundle either: the fetch fails as before
            pass
    return context


#: how long File -> Check for Updates waits, in seconds; it is the one
#: place the application asks. Longer than `TIMEOUT`, which stays the
#: default for scripted `fetch` and `check` calls, because somebody who
#: chose the menu item is waiting on an answer and a corporate network
#: is not always quick.
ASKED_TIMEOUT = 12.0


def reason_for(error: BaseException, timeout: float) -> str:
    """Why the manifest could not be read, in one sentence for a status
    line.

    The sentence names the cause instead of the exception: an HTTP
    status, a certificate that could not be verified, no answer within
    `timeout`, an answer that is not a readable manifest, or a host that
    could not be reached. A proxy, a filtered domain, an unverifiable
    certificate, a slow network and a site that is down cannot be told
    apart from outside the program, so the message is what tells them
    apart.

    `urlopen` wraps a refused connection, a timeout and a certificate
    that will not verify in a `URLError` whose `reason` is the real
    exception, so that is unwrapped first. `HTTPError` is a `URLError`
    too, and is read before unwrapping because its status is the news.

    Parameters
    ----------
    error : BaseException
        What the fetch raised.
    timeout : float
        The timeout it was given, in seconds, for the message.

    Returns
    -------
    str
        The reason, in words.
    """
    # Folding every failure into "could not reach visualdynamics.org"
    # was the original behavior, and it cost an afternoon of diagnosis
    # by hand on a work machine (Brandon, 2026-09-23).
    #
    # The wrapping was measured against a self-signed host (2026-09-23).
    # The first version tested for the certificate error before
    # unwrapping, so a real one never matched and read as "could not be
    # reached", the one message that hides the likeliest cause on an
    # inspecting network. Its test passed only because it raised the
    # certificate error bare, a shape urllib never produces.
    from urllib.error import HTTPError, URLError

    if isinstance(error, HTTPError):
        return (f'visualdynamics.org answered {error.code} '
                f'{error.reason} — something between here and it is '
                'refusing the request')
    if isinstance(error, URLError):
        error = error.reason
    if isinstance(error, ssl.SSLCertVerificationError):
        # every one of these is optional on the exception — ssl fills
        # them in, a hand-raised one may not, and a status line is no
        # place to raise AttributeError
        said = (getattr(error, 'verify_message', None)
                or getattr(error, 'reason', None) or error)
        return (f'visualdynamics.org could not be verified: {said}. A '
                'network that inspects HTTPS re-signs it with its own '
                'authority; the check trusts what this machine trusts, '
                'so that authority has to be installed here')
    if isinstance(error, TimeoutError):      # socket.timeout is this alias
        return f'visualdynamics.org did not answer within {timeout:g} s'
    if isinstance(error, (ValueError, UnicodeDecodeError)):
        return 'visualdynamics.org answered, but not with a readable manifest'
    if isinstance(error, (OSError, str)):    # a URLError's reason is either
        return f'visualdynamics.org could not be reached: {error}'
    return f'the update check failed: {type(error).__name__}: {error}'


def attempt(url: str = MANIFEST,
            timeout: float = TIMEOUT) -> tuple[dict[str, Any] | None, str | None]:
    """Read the manifest, and say why not when it cannot be read.

    This is what File > Check for Updates calls, and `fetch` is built on
    it. It reads at most 64 kB, and every failure is returned as a
    reason (`reason_for`), never raised, so a person waiting on the menu
    item is told which thing went wrong.

    Parameters
    ----------
    url : str, optional
        Where the manifest is; `MANIFEST` by default.
    timeout : float, optional
        How long to wait, in seconds; `TIMEOUT` (4 s) by default.

    Returns
    -------
    tuple
        ``(manifest, None)`` with the manifest as a dict, or
        ``(None, reason)`` with the reason as a string.
    """
    try:
        request = Request(url, headers={
            'User-Agent': f'VisualDynamics/{__version__}'})
        with urlopen(request, timeout=timeout,
                     context=trust_store()) as response:
            manifest = json.loads(response.read(64_000).decode('utf-8'))
    except Exception as error:   # noqa: BLE001 — every failure here is
        return None, reason_for(error, timeout)   # news, never a
                                                 # traceback in front
                                                 # of somebody who
                                                 # only opened the app
    if not isinstance(manifest, dict):
        return None, 'visualdynamics.org answered, but not with a manifest'
    return manifest, None


def fetch(url: str = MANIFEST, timeout: float = TIMEOUT) -> dict[str, Any] | None:
    """The manifest as published, or None when it cannot be read.

    The same request as `attempt`, without the reason. Use `attempt`
    where "up to date" and "could not ask" need different words, as
    they do in the app's menu item.

    Parameters
    ----------
    url : str, optional
        Where the manifest is; `MANIFEST` by default.
    timeout : float, optional
        How long to wait, in seconds; `TIMEOUT` (4 s) by default.

    Returns
    -------
    dict or None
        The manifest, or None.
    """
    return attempt(url, timeout)[0]


def check(url: str = MANIFEST, timeout: float = TIMEOUT) -> dict[str, Any] | None:
    """The manifest when it offers a newer version, else None.

    For a script that only wants to know whether to act. Every failure
    is None as well: no network, a captive portal, a 404, a truncated
    file, a version that does not parse. The request blocks for up to
    `timeout` seconds, so a caller with a window to keep responsive runs
    it off the main thread.

    Parameters
    ----------
    url : str, optional
        Where the manifest is; `MANIFEST` by default.
    timeout : float, optional
        How long to wait, in seconds; `TIMEOUT` (4 s) by default.

    Returns
    -------
    dict or None
        The manifest, when its version is later than the one running.
    """
    manifest = fetch(url, timeout)
    if manifest is None:
        return None
    version = str(manifest.get('version', ''))
    return manifest if version and newer(version) else None


def upgrade_command() -> str | None:
    """The command that updates this copy, when it is a pip install.

    The command is spelled with this interpreter's own path, so it
    updates this environment and not whichever `pip` the shell finds
    first. The app shows it with a button that copies it; it never runs
    it.

    Returns
    -------
    str or None
        ``<python> -m pip install --upgrade visualdynamics``, or None
        for the packaged application (a download, or Sparkle on macOS),
        for a source tree that is not installed, and for an editable
        install, where `pip install --upgrade` would replace the working
        tree with the PyPI release and `git pull` is the update.
    """
    # A pip install downloads only the package, a few MB, not a 480 MB
    # application, so the news that a version is available comes with
    # the command to take it (Brandon, 2026-09-28: show the command, with
    # a copy button; the app does not run it).
    import importlib.metadata
    import shlex
    import sys

    if getattr(sys, 'frozen', False):
        return None
    try:
        distribution = importlib.metadata.distribution('visualdynamics')
    except importlib.metadata.PackageNotFoundError:
        return None                    # run from a source tree, uninstalled
    try:
        direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    except ValueError:
        direct = {}
    if direct.get('dir_info', {}).get('editable'):
        return None
    python = (f'"{sys.executable}"' if sys.platform == 'win32'
              else shlex.quote(sys.executable))
    return f'{python} -m pip install --upgrade visualdynamics'
