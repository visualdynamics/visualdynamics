"""Is there a newer version than this one?

A check, not an updater. It asks one URL for one small JSON file, and
the answer goes in a status line: *0.1.0 is available*. Nothing is
downloaded, nothing is replaced, nothing runs with privileges.

That is a deliberate stopping point. A real auto-updater has to verify
what it downloaded before executing it, which means signing keys and a
threat model — an unverified one is a remote code execution feature
with a friendly name. Sparkle (macOS) and WinSparkle do it properly and
can be adopted later; until there is a signing identity to verify
against, telling the user beats pretending.

The manifest lives on `visualdynamics.org`, a static host with no
server to run — the release workflow writes it from each published
release and deploys it with the site (`web/public/latest.json` is the
holding page's copy). Its shape:

    {"version": "0.1.0", "url": "https://…/releases", "notes": "…"}
"""

from __future__ import annotations

import json
import ssl
from typing import Any
from urllib.request import Request, urlopen

from . import __version__

MANIFEST = 'https://visualdynamics.org/latest.json'
TIMEOUT = 4.0


def parse_version(text: str) -> tuple[int, ...]:
    """'0.10.2' -> (0, 10, 2, 1, 0), for comparing rather than for showing.

    Numeric, so 0.10 sorts above 0.9 — a string compare puts them the
    other way round and would offer a downgrade as an upgrade. Anything
    unparseable becomes zero rather than raising: a malformed manifest
    should mean *no update offered*, never a traceback in front of
    somebody who only opened the application.

    A pre-release sorts *below* its release: '0.1.0a1' is (0, 1, 0, 0,
    1) and '0.1.0' is (0, 1, 0, 3, 0), so the first non-alpha 0.1.0 is
    offered to an alpha as the upgrade it is (2026-09-11). The stage is
    a for alpha, b for beta, rc for a candidate — Python's own
    spelling, and the one the version string carries — and a release
    is stage 3, above them all.
    """
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
    """Is `available` a later version than what is running?"""
    return parse_version(available) > parse_version(running)


def trust_store() -> ssl.SSLContext:
    """The TLS context the check verifies visualdynamics.org with: what
    the operating system trusts.

    `truststore` (MIT; what pip itself uses) verifies against the
    system's own store — the Keychain on macOS, the certificate store
    on Windows, OpenSSL's system bundle on Linux. That is where a
    corporate network's own certificate authority lives: a network that
    inspects HTTPS re-signs every site with it, IT installs it on the
    machine, and `curl` and the browser trust it while a Python bundle
    of Mozilla's roots has never heard of it. Brandon's work machine
    reported exactly that, "unable to get local issuer certificate", on
    a network where `curl` got a 200 (2026-09-25).

    Without `truststore` — a bare checkout without it installed — the
    older arrangement stands: a packaged build carries its own OpenSSL,
    and the one the wheels bring was built elsewhere with a compiled-in
    certificate directory that exists on nobody else's machine (read
    off the library with `strings`, 2026-09-15), so the default context
    trusts nothing; when it is empty the Mozilla bundle certifi ships
    is loaded instead.
    """
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


#: how long File -> Check for Updates waits, the one place the
#: application asks. Longer than `TIMEOUT`, which stays the default for
#: scripted `fetch` and `check` calls: somebody who chose the menu item
#: is waiting on an answer, and a corporate network is not always quick.
ASKED_TIMEOUT = 12.0


def reason_for(error: BaseException, timeout: float) -> str:
    """Why the manifest could not be had, in words for a status line.

    One sentence, naming the cause rather than the exception. Folding
    every failure into "could not reach visualdynamics.org" was the
    original behavior and it cost a long diagnosis by hand: a proxy, a
    filtered domain, an unverifiable certificate, a slow network and a
    site that is down all read identically, and none of them can be
    told apart from outside the program (Brandon, 2026-09-23).

    **`urlopen` wraps the cause.** A refused connection, a timeout and a
    certificate that will not verify all arrive as a `URLError` whose
    `reason` is the real exception — measured against a self-signed
    host, 2026-09-23. The first version tested for the certificate
    error *before* unwrapping, so a real one never matched and read as
    "could not be reached", which is the one message that hides the
    likeliest cause on an inspecting network. Its test passed only
    because it raised the certificate error bare, a shape urllib never
    produces. So `HTTPError` first (it *is* a `URLError`, and its
    status is the news), then unwrap, then read the cause once.
    """
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
    """(manifest, None), or (None, why not) in words.

    The reading `fetch` is built on. It exists because a status line
    that says only "could not reach visualdynamics.org" cannot be
    acted on: the answer a person needs is *which* of the half-dozen
    things went wrong.
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
    """The manifest as published, or None when it cannot be had.

    `check` folds "nothing newer" and "could not ask" into one answer
    for a script that only wants to know whether to act. Where the two
    deserve different words — up to date is not the same news as
    offline — `attempt` carries the reason, and the menu uses that.
    """
    return attempt(url, timeout)[0]


def check(url: str = MANIFEST, timeout: float = TIMEOUT) -> dict[str, Any] | None:
    """The manifest when it offers something newer, else None.

    Every failure is None: no network, a captive portal, a 404, a
    truncated file, a version that does not parse. An update check is
    the least important thing the application does and must never be
    the reason it is slow to start or noisy on launch — which is why
    this takes a timeout and why the caller runs it off the main
    thread.
    """
    manifest = fetch(url, timeout)
    if manifest is None:
        return None
    version = str(manifest.get('version', ''))
    return manifest if version and newer(version) else None
