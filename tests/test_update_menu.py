"""File → Check for Updates: the answer in words, never an action.

The window asks `update.attempt` off the main thread and says one of
three things: a newer version is available (with the page to open),
this one is up to date, or **why** it could not ask — offline is not
the same news as current, and a proxy is not the same news as an
unverifiable certificate. Nothing is downloaded and nothing runs;
`update.py` explains why.
"""

from __future__ import annotations

import ssl
import sys

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from visualdynamics import __version__, update


def _settled(window, pump):
    """The status line once the check's thread has answered."""
    for _ in range(50):
        QTest.qWait(20)
        if 'Checking' not in window.statusBar().currentMessage():
            break
    pump(3)
    return window.statusBar().currentMessage()


def _answer(window, pump, monkeypatch, manifest, why=None):
    monkeypatch.setattr(update, 'attempt', lambda *a, **k: (manifest, why))
    window.check_for_updates()
    return _settled(window, pump)


def test_the_menu_offers_it(window):
    labels = [a.text() for a in window.menuBar().actions()
              for a in a.menu().actions()]
    assert 'Check for &Updates...' in labels


def test_offline_is_said_as_offline(window, pump, monkeypatch):
    said = _answer(window, pump, monkeypatch, None,
                   why='visualdynamics.org could not be reached: '
                       'nodename nor servname provided')
    assert 'could not be reached' in said
    assert 'up to date' not in said, 'offline must never read as current'


def test_the_same_version_is_up_to_date(window, pump, monkeypatch):
    said = _answer(window, pump, monkeypatch, {'version': __version__})
    assert 'up to date' in said
    assert __version__ in said


def test_a_newer_version_is_offered_as_a_page_to_open(window, pump,
                                                       monkeypatch):
    opened = []
    from PySide6.QtGui import QDesktopServices
    monkeypatch.setattr(QDesktopServices, 'openUrl',
                        staticmethod(lambda url: opened.append(url.toString())))
    said = _answer(window, pump, monkeypatch, {
        'version': '99.0.0', 'url': 'https://visualdynamics.org/get',
        'notes': 'Everything works now.'})
    assert '99.0.0 is available' in said
    box = window.findChild(QMessageBox)
    assert box is not None and box.isVisible(), 'the offer is a box to act on'
    assert '99.0.0' in box.text() and __version__ in box.text()
    assert box.informativeText() == 'Everything works now.'
    opener = next(b for b in box.buttons() if 'Open' in b.text())
    opener.click()
    pump(3)
    assert opened == ['https://visualdynamics.org/get'], (
        'the page opens in the browser; nothing is downloaded here')


def test_the_reason_is_what_reaches_the_status_line(window, pump, monkeypatch):
    """Not "could not reach" but *why*.

    A work machine that could not check for updates cost an afternoon
    of commands to diagnose, and the answer turned out not to be on
    that machine at all (Brandon, 2026-09-23). Every failure reading
    the same is what made that necessary: a proxy, a filtered domain,
    an unverifiable certificate, a slow network and a site that is
    down are five different problems and one sentence.
    """
    said = _answer(window, pump, monkeypatch, None,
                   why='visualdynamics.org answered 403 Forbidden — '
                       'something between here and it is refusing the '
                       'request')
    assert '403' in said and 'refusing' in said
    assert 'up to date' not in said


def test_the_asked_for_check_is_the_patient_one(window, pump, monkeypatch):
    """Somebody who chose the menu item is waiting on an answer, and a
    corporate network is not always quick.

    Read off the call, not off the source. The first version of this
    scanned the function's text for the name, which its docstring
    mentions too — so removing the argument itself left the test green
    (2026-09-23).
    """
    assert update.ASKED_TIMEOUT > update.TIMEOUT
    asked = {}

    def record(*args, **kwargs):
        asked['timeout'] = kwargs.get(
            'timeout', args[1] if len(args) > 1 else None)
        return {'version': __version__}, None

    monkeypatch.setattr(update, 'attempt', record)
    window.check_for_updates()
    _settled(window, pump)
    assert asked['timeout'] == update.ASKED_TIMEOUT, (
        f'the menu asked with {asked["timeout"]}, not the patient deadline'
    )


def test_the_check_trusts_what_the_machine_trusts():
    """The context verifies against the operating system's own store,
    where a corporate network's inspecting authority is installed and
    where a Python bundle of Mozilla's roots never looks. Brandon's work
    machine failed on exactly that, "unable to get local issuer
    certificate", while curl and the browser trusted the same site
    (2026-09-25)."""
    truststore = pytest.importorskip('truststore')
    context = update.trust_store()
    assert isinstance(context, truststore.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_without_truststore_the_older_arrangement_stands(monkeypatch):
    """A checkout without the package still verifies: the default
    context, or the Mozilla bundle when the packaged OpenSSL's store
    is empty."""
    monkeypatch.setitem(sys.modules, 'truststore', None)
    context = update.trust_store()
    assert type(context) is ssl.SSLContext
    assert context.get_ca_certs(), 'a store with something in it'


def test_each_failure_is_named_by_its_cause():
    """`reason_for` maps an exception to a sentence a person can act
    on, checked on the **shapes urllib actually raises**.

    `urlopen` wraps a refused connection, a timeout and a certificate
    that will not verify in a `URLError` whose `reason` is the real
    exception (measured against a self-signed host, 2026-09-23). The
    first version of this test raised the certificate error *bare*, a
    shape urllib never produces, and passed while a real certificate
    failure read as "could not be reached" — the one message that hides
    the likeliest cause on a network that inspects HTTPS.
    """
    import ssl
    from urllib.error import HTTPError, URLError

    said = update.reason_for(
        HTTPError('u', 403, 'Forbidden', {}, None), 4.0)
    assert '403' in said and 'Forbidden' in said

    certificate = ssl.SSLCertVerificationError(
        'unable to get local issuer certificate')
    said = update.reason_for(URLError(certificate), 4.0)
    assert 'verified' in said and 'inspects HTTPS' in said, (
        'a certificate failure, as urlopen really raises it, is named as one'
    )
    assert 'could not be reached' not in said

    said = update.reason_for(URLError(TimeoutError('timed out')), 9.0)
    assert 'did not answer within 9 s' in said

    said = update.reason_for(URLError(ConnectionRefusedError('refused')), 4.0)
    assert 'could not be reached' in said

    said = update.reason_for(URLError('nodename nor servname provided'), 4.0)
    assert 'could not be reached' in said, 'a reason can also be a string'

    said = update.reason_for(ValueError('no json'), 4.0)
    assert 'readable manifest' in said

    said = update.reason_for(RuntimeError('something else'), 4.0)
    assert 'RuntimeError' in said, 'an unexpected failure still names itself'


def test_the_fetch_still_answers_with_a_manifest_or_none(monkeypatch):
    """`fetch` keeps its shape, built on `attempt`: the manifest, or
    None where `attempt` has a reason. Driven without the network — a
    real lookup of an unresolvable name waits on the system resolver,
    which on a slow or filtered network is seconds on every gate run."""
    import inspect
    from urllib.error import URLError

    assert inspect.signature(update.fetch).parameters.keys() == {
        'url', 'timeout'}

    def unreachable(*args, **kwargs):
        raise URLError(ConnectionRefusedError('refused'))

    monkeypatch.setattr(update, 'urlopen', unreachable)
    manifest, why = update.attempt()
    assert manifest is None and 'could not be reached' in why
    assert update.fetch() is None
