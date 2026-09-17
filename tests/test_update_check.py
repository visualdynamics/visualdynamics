"""The update check: an answer, never an action.

It fetches one small JSON file and reports whether a newer version
exists. It does not download, replace, or execute anything — that
boundary is the whole design, because an updater that runs what it
fetched without verifying a signature is a remote code execution
feature with a friendly name.

So what is tested here is mostly *refusal*: every way the check can go
wrong has to come back as "no update", quietly, on a timeout, off the
main thread — never as a traceback in front of somebody who opened the
application to look at a mode shape.
"""

from __future__ import annotations

import json

import pytest

from visualdynamics import __version__
from visualdynamics.update import check, fetch, newer, parse_version


def test_ten_is_after_nine():
    """The bug a string comparison guarantees, and the reason this
    parses to integers: '0.10.0' < '0.9.0' as text, so a string compare
    would offer 0.9.0 to somebody already running 0.10.0 — an upgrade
    button that downgrades."""
    assert newer('0.10.0', '0.9.0')
    assert not newer('0.9.0', '0.10.0')
    assert parse_version('0.10.0') > parse_version('0.9.0')


def test_the_running_version_is_not_newer_than_itself():
    assert not newer(__version__, __version__)


@pytest.mark.parametrize('text', ['', 'not-a-version', '..', 'v', None])
def test_nonsense_parses_to_something_ordered_rather_than_raising(text):
    """A malformed manifest means no update offered. It must not mean a
    crash, and it must not mean an update offered by accident."""
    assert parse_version(str(text)) <= parse_version(__version__) or True
    assert not newer(str(text), '99.0.0')


def manifest_server(body, status=200, delay=0.0):
    """A one-request localhost server standing in for the real host."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(delay)
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    return f'http://127.0.0.1:{server.server_port}/latest.json'


def test_a_newer_version_comes_back_whole():
    """The caller wants the notes and the link, not just a yes."""
    url = manifest_server(json.dumps({
        'version': '99.0.0', 'url': 'https://visualdynamics.org/releases',
        'notes': 'Everything works now.'}).encode())
    found = check(url)
    assert found is not None
    assert found['version'] == '99.0.0'
    assert found['notes'] == 'Everything works now.'


def test_the_same_version_is_not_an_update():
    url = manifest_server(json.dumps({'version': __version__}).encode())
    assert check(url) is None


@pytest.mark.parametrize('body,status', [
    (b'{"version": "99.0.0"', 200),      # truncated JSON
    (b'<html>404</html>', 404),          # a host that lost the file
    (b'{}', 200),                        # valid JSON, no version
    (b'{"version": ""}', 200),           # a version that says nothing
])
def test_every_bad_answer_is_no_update_rather_than_an_exception(body, status):
    assert check(manifest_server(body, status)) is None


def test_an_unreachable_host_is_no_update():
    """No network, a captive portal, a firewall. The commonest case on a
    laptop, and the one that must be silent."""
    assert check('http://127.0.0.1:1/latest.json', timeout=1.0) is None


def test_a_slow_host_gives_up_rather_than_hanging():
    """The check is the least important thing the application does and
    must never be why it is slow."""
    import time

    url = manifest_server(json.dumps({'version': '99.0.0'}).encode(),
                          delay=3.0)
    started = time.monotonic()
    assert check(url, timeout=0.5) is None
    assert time.monotonic() - started < 2.0, 'the timeout was not honored'


def test_fetch_tells_offline_from_up_to_date():
    """`check` folds the two into "no update", right for a silent
    check; a menu that was *asked* owes different words for each."""
    assert fetch('http://127.0.0.1:1/latest.json', timeout=1.0) is None
    url = manifest_server(json.dumps({'version': __version__}).encode())
    assert fetch(url) == {'version': __version__}
    assert fetch(manifest_server(b'[1, 2, 3]')) is None, (
        'a manifest that is not a mapping is no manifest')


def test_the_check_trusts_certifi_when_the_packaged_openssl_trusts_nothing():
    """The wheels' OpenSSL was built with a certificate directory that
    exists on nobody's machine, so the default context trusts nothing
    and every HTTPS request failed: "could not reach visualdynamics.org"
    on a Mac that was online (Brandon, 2026-09-15). When the default
    store is empty the Mozilla bundle certifi ships is loaded; when it
    is not, the platform's own store is left alone."""
    import ssl

    from visualdynamics import update

    class Bare:
        def __init__(self):
            self.loaded = []

        def get_ca_certs(self):
            return [{'subject': 'x'}] if self.loaded else []

        def load_verify_locations(self, path):
            self.loaded.append(path)

    empty = Bare()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ssl, 'create_default_context', lambda: empty)
        context = update.trust_store()
    assert context is empty
    assert len(empty.loaded) == 1 and empty.loaded[0].endswith('cacert.pem')

    full = Bare()
    full.loaded.append('platform')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ssl, 'create_default_context', lambda: full)
        update.trust_store()
    assert full.loaded == ['platform'], 'a store that has certificates is kept'


def test_an_empty_store_with_no_certifi_is_returned_not_raised():
    """A packaged copy that somehow lacks the bundle too: the fetch then
    fails as it did before, reported as unreachable — never as a
    traceback out of the menu."""
    import ssl
    import sys

    from visualdynamics import update

    class Bare:
        def __init__(self):
            self.loaded = []

        def get_ca_certs(self):
            return []

        def load_verify_locations(self, path):
            self.loaded.append(path)

    empty = Bare()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ssl, 'create_default_context', lambda: empty)
        patch.setitem(sys.modules, 'certifi', None)   # `import certifi` raises
        assert update.trust_store() is empty
    assert empty.loaded == []


def test_a_fetch_with_no_platform_store_still_reaches_the_site():
    """The same condition end to end, in a fresh process: OpenSSL told
    to look for certificates where there are none. Skipped offline."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, SSL_CERT_FILE='/nonexistent',
               SSL_CERT_DIR='/nonexistent')
    probe = ('import json, sys\n'
             'from visualdynamics import update\n'
             'print(json.dumps(update.fetch(timeout=6.0)))\n')
    out = subprocess.run([sys.executable, '-c', probe], env=env,
                         capture_output=True, text=True, timeout=30,
                         check=False)
    assert out.returncode == 0, out.stderr[-400:]
    manifest = json.loads(out.stdout.strip())
    if manifest is None:
        online = subprocess.run(
            [sys.executable, '-c', ('from visualdynamics import update; '
                                    'import json; print(json.dumps('
                                    'update.fetch(timeout=6.0)))')],
            capture_output=True, text=True, timeout=30,
            check=False).stdout.strip()
        if json.loads(online) is None:
            pytest.skip('visualdynamics.org is not reachable from here')
        pytest.fail('reachable with the platform store, not without it: '
                    'the certifi fallback did not engage')
    assert 'version' in manifest
