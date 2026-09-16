"""The packaging scripts' one testable decision each.

`build_macos.sh` decides how to notarise: from a login-keychain item
when one holds a password, else from the notarytool profile. The
item was created empty once (2026-09-15: the command lacked its
trailing `-w`), and the script's answer to an empty item is what kept
that build notarising. The function is lifted out of the script and
run against a stub `security` on PATH — the script itself is not
sourceable (it signs and wipes dist/ at the top level).

`smoke_windows.ps1` runs only on Windows; what can be held here is
that the traceback check is still in it and the release still runs
it — the first Windows launch of 0.1.0a1 crashed behind a titled
window and the test called it a pass (2026-09-14).
"""

from __future__ import annotations

import os
import re
import stat
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
SCRIPT = os.path.join(ROOT, 'packaging', 'build_macos.sh')


def notary_args(tmp_path, security: str) -> list[str]:
    with open(SCRIPT, encoding='utf-8') as handle:
        text = handle.read()
    start = text.index('NOTARY_PROFILE=')
    end = text.index('notary_ready()')
    function = text[start:end]
    stub = tmp_path / 'security'
    stub.write_text('#!/bin/bash\n' + security)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    out = subprocess.run(
        ['bash', '-c', function + '\nnotary_args > /dev/null; '
         'printf "%s\\n" "${NOTARY[@]}"'],
        capture_output=True, text=True, check=True,
        env={**os.environ, 'PATH': f'{tmp_path}:{os.environ["PATH"]}'})
    return out.stdout.split('\n')[:-1]


@pytest.mark.skipif(os.name != 'posix', reason='a bash function')
def test_an_empty_keychain_item_falls_back_to_the_profile(tmp_path):
    assert notary_args(tmp_path, 'exit 0\n') == [
        '--keychain-profile', 'vd-notary']
    assert notary_args(tmp_path, 'exit 44\n') == [
        '--keychain-profile', 'vd-notary'], 'no item at all'


@pytest.mark.skipif(os.name != 'posix', reason='a bash function')
def test_a_keychain_item_with_a_password_notarises_as_its_account(tmp_path):
    security = ('case " $* " in *" -w "*) echo "s3cret";; '
                '*) echo \'    "acct"<blob>="someone@example.com"\';; esac\n')
    assert notary_args(tmp_path, security) == [
        '--apple-id', 'someone@example.com', '--team-id', '2542NQ9D95',
        '--password', 's3cret']


def test_the_windows_smoke_test_fails_on_a_traceback_and_the_release_runs_it():
    with open(os.path.join(ROOT, 'packaging', 'smoke_windows.ps1'),
              encoding='utf-8') as handle:
        script = handle.read()
    check = re.search(r"Select-String -Path \$err -Pattern '([^']*)'", script)
    assert check and 'Traceback' in check.group(1)
    after = script[check.end():]
    assert after.lstrip().startswith('-Quiet)) {') and 'throw' in after[:400]
    assert '--no-disclaimer' in script, 'launched past the alpha notice'
    with open(os.path.join(ROOT, '.github', 'workflows', 'release.yml'),
              encoding='utf-8') as handle:
        workflow = handle.read()
    assert 'packaging\\smoke_windows.ps1' in workflow
