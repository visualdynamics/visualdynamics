"""The packaging scripts' one testable decision each.

`build_macos.sh` decides how to notarize: from a login-keychain item
when one holds a password, else from the notarytool profile. The
item was created empty once (2026-09-15: the command lacked its
trailing `-w`), and the script's answer to an empty item is what kept
that build notarizing. The function is lifted out of the script and
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


def notary_args(tmp_path, security: str, **env) -> list[str]:
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
        env={**os.environ, 'PATH': f'{tmp_path}:{os.environ["PATH"]}', **env})
    return out.stdout.split('\n')[:-1]


@pytest.mark.skipif(os.name != 'posix', reason='a bash function')
def test_an_empty_keychain_item_falls_back_to_the_profile(tmp_path):
    assert notary_args(tmp_path, 'exit 0\n') == [
        '--keychain-profile', 'vd-notary']
    assert notary_args(tmp_path, 'exit 44\n') == [
        '--keychain-profile', 'vd-notary'], 'no item at all'


@pytest.mark.skipif(os.name != 'posix', reason='a bash function')
def test_a_keychain_item_with_a_password_notarizes_as_its_account(tmp_path):
    security = ('case " $* " in *" -w "*) echo "s3cret";; '
                '*) echo \'    "acct"<blob>="someone@example.com"\';; esac\n')
    assert notary_args(tmp_path, security) == [
        '--apple-id', 'someone@example.com', '--team-id', '2542NQ9D95',
        '--password', 's3cret']


@pytest.mark.skipif(os.name != 'posix', reason='a bash function')
def test_a_runner_notarizes_from_its_secrets_ahead_of_the_keychain(tmp_path):
    """The release workflow hands the Apple ID and its app-specific
    password in through the environment (2026-10-10); they win over a
    keychain item, which a runner has not got anyway."""
    security = ('case " $* " in *" -w "*) echo "s3cret";; '
                '*) echo \'    "acct"<blob>="someone@example.com"\';; esac\n')
    assert notary_args(tmp_path, security, NOTARY_APPLE_ID='ci@example.com',
                       NOTARY_PASSWORD='from-a-secret') == [
        '--apple-id', 'ci@example.com', '--team-id', '2542NQ9D95',
        '--password', 'from-a-secret']
    assert notary_args(tmp_path, security, NOTARY_APPLE_ID='ci@example.com',
                       NOTARY_PASSWORD='')[:2] == [
        '--apple-id', 'someone@example.com'], 'half a secret is none'


@pytest.mark.skipif(os.name != 'posix', reason='a bash function')
def test_a_release_build_that_cannot_notarize_stops_before_building(tmp_path):
    """On a tag the workflow sets VD_REQUIRE_NOTARIZED, and a build with
    no identity or no working credentials stops before PyInstaller runs
    rather than shipping an image Gatekeeper refuses; unset, the same
    build goes on to make its ad-hoc image."""
    with open(SCRIPT, encoding='utf-8') as handle:
        text = handle.read()
    start = text.index('NOTARY_PROFILE=')
    end = text.index('rm -rf build dist')
    stub = tmp_path / 'security'
    stub.write_text('#!/bin/bash\nexit 44\n')
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)

    def run(**env):
        return subprocess.run(
            ['bash', '-c', 'SIGN=\n' + text[start:end] + '\necho went-on'],
            check=False, capture_output=True, text=True,
            env={**os.environ, 'PATH': f'{tmp_path}:{os.environ["PATH"]}', **env})

    refused = run(VD_REQUIRE_NOTARIZED='1')
    assert refused.returncode == 1 and 'went-on' not in refused.stdout
    assert 'must be signed and notarized' in refused.stderr
    allowed = run(VD_REQUIRE_NOTARIZED='')
    assert allowed.returncode == 0 and 'went-on' in allowed.stdout


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


def test_both_packaged_builds_check_they_can_save_an_animation():
    """The bundle carrying QtMultimedia proves nothing about the encoder
    answering; each packaged build writes and reads back a movie."""
    with open(os.path.join(ROOT, 'packaging', 'smoke_windows.ps1'),
              encoding='utf-8') as handle:
        smoke = handle.read()
    assert "'--check-movie'" in smoke and 'throw "the movie check failed' in smoke
    assert smoke.index('--check-movie') < smoke.index('# 2. A small real project')
    with open(os.path.join(ROOT, 'packaging', 'build_macos.sh'),
              encoding='utf-8') as handle:
        build = handle.read()
    line = next(line for line in build.splitlines() if '--check-movie' in line)
    assert line.startswith('"$APP/Contents/MacOS/Visual Dynamics"')
    assert build.index('--check-movie') < build.index('if [[ -n $SIGN ]]'), \
        'checked before it is signed and shipped'


def test_the_venvs_are_levelled_eagerly_and_gated_by_pytest_s_own_code():
    """`tools/level_venv.sh` is step 0 of a release. A plain `pip install
    -U` upgrades the package and keeps every dependency it has, which is
    how this desk drifted behind CI (2026-10-09); and the gate is judged
    by pytest's exit code, never through a pipe."""
    with open(os.path.join(ROOT, 'tools', 'level_venv.sh'),
              encoding='utf-8') as handle:
        script = handle.read()
    assert 'pip install -q -U --upgrade-strategy eager' in script
    assert "level .venv -e '.[dev,step,docs,app]'" in script
    assert "level .venv-x86_64 '.[step,app]'" in script
    gate = next(line for line in script.splitlines()
                if '-m pytest tests' in line)
    assert '|' not in gate and gate.rstrip().endswith('\\')
    assert 'exit $code' in script
    with open(os.path.join(ROOT, 'REMAINING-TASKS.md'),
              encoding='utf-8') as handle:
        assert 'tools/level_venv.sh' in handle.read()
