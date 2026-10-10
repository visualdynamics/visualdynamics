"""The launch site's preview hands out what this machine built.

`packaging/stage_downloads.py` hardlinks the installers in `dist/`
into `web/launch/downloads/` (gitignored, cleared before any deploy)
and writes `manifest.json` beside them **in the GitHub releases API's
own shape** — so `downloads.html` fills its three slots with one
routine whether the source is the staged manifest (localhost) or the
released assets (deployed).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'packaging'))

from stage_downloads import manifest_for, version_of


def test_the_manifest_speaks_the_releases_apis_shape():
    manifest = manifest_for([
        ('VisualDynamics-0.0.1-windows-x64-setup.exe', 261 * 2**20),
        ('VisualDynamics-0.0.1.dmg', 415 * 2**20)])
    assert manifest['local'] is True
    names = [a['name'] for a in manifest['assets']]
    assert names == sorted(names), 'stable order, whatever dist listed'
    windows = next(a for a in manifest['assets']
                   if a['name'].endswith('setup.exe'))
    # the keys the page reads off a GitHub release asset, exactly
    assert windows['browser_download_url'] \
        == 'downloads/VisualDynamics-0.0.1-windows-x64-setup.exe'
    assert windows['size'] == 261 * 2**20


def test_nothing_staged_is_an_empty_manifest_not_a_missing_one():
    manifest = manifest_for([])
    assert manifest['assets'] == [], (
        'the page falls back to the releases API on an empty list')


def test_the_page_asks_for_the_manifest_only_locally():
    """The deployed page must never request a file that only exists on
    a development machine — the guard is the hostname check."""
    import pathlib

    page = pathlib.Path(os.path.dirname(__file__), '..', 'web', 'launch',
                        'downloads.html').read_text(encoding='utf-8')
    assert "fetch('downloads/manifest.json')" in page
    assert "here === 'localhost' || here === '127.0.0.1'" in page
    assert page.index("localhost") < page.index(
        "fetch('downloads/manifest.json')"), (
        'the hostname gate reads before the fetch it guards')


def test_the_site_job_deploys_the_directory_the_docs_build_into():
    """One deploy carries the launch pages, the documentation and the
    update manifest (Brandon, 2026-09-01). That holds only while the
    workflow deploys the directory mkdocs builds into, and only while
    the job runs on a *published* release — a tag whose release is
    still a draft must not put a manifest in front of the update
    check that names assets nobody can download yet."""
    import os
    import posixpath

    import yaml

    root = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(root, '.github', 'workflows', 'release.yml'),
              encoding='utf-8') as handle:
        workflow = yaml.safe_load(handle)
    with open(os.path.join(root, 'properdocs.yml'), encoding='utf-8') as handle:
        docs = yaml.safe_load(handle)
    assert workflow[True]['release'] == {'types': ['published']}
    site = workflow['jobs']['site']
    gate = ' '.join(site['if'].split())
    assert ("github.event_name == 'release' && "
            "startsWith(github.event.release.tag_name, 'v')") in gate, \
        'the site job runs for a version release only (2026-09-14)'
    assert "github.event_name == 'release' &&" not in gate.replace(
        "(github.event_name == 'release' &&", ''), 'the tag check is not optional'
    deploy = [step for step in site['steps'] if 'wrangler' in step.get('run', '')]
    assert len(deploy) == 1
    # the step runs inside the site directory and deploys '.', so that
    # wrangler finds ./functions where it runs (2026-09-14) — the
    # directory it deploys is the two joined
    target = deploy[0]['run'].split('pages deploy ')[1].split()[0]
    # posixpath, not os.path: both halves come out of YAML written with
    # forward slashes, and the comparison below is against `site_dir`,
    # which is also YAML. On Windows os.path.join would hand back
    # `web\launch` and the test would fail on the separator rather than
    # on anything about the workflow (Kevin Cross, 2026-09-21).
    deployed = posixpath.normpath(
        posixpath.join(deploy[0].get('working-directory', '.'), target))
    assert docs['site_dir'].startswith(deployed + '/'), (
        f'mkdocs builds into {docs["site_dir"]}, the job deploys {deployed}')
    manifest = [step for step in site['steps'] if 'latest.json' in step.get('run', '')]
    assert manifest and f"{deployed}/latest.json" in manifest[0]['run']
    for name in ('linux', 'windows', 'macos'):
        assert "github.event_name != 'release'" in workflow['jobs'][name]['if'], (
            f'{name} would rebuild on every publish')


def test_the_manifest_is_one_release_the_newest():
    """Staging is cumulative, so two versions can sit in the folder;
    the page takes the first file matching a platform, and a
    name-sorted list handed out 0.0.1 ahead of 0.1.0 — the old,
    unsigned image, the evening the notarized one landed (Brandon,
    2026-09-02). A manifest is one release: the newest version's
    files, and nothing older."""
    manifest = manifest_for([
        ('VisualDynamics-0.0.1-macos-arm64.dmg', 10),
        ('VisualDynamics-0.1.0-macos-arm64.dmg', 11),
        ('VisualDynamics-0.1.0-windows-x64-setup.exe', 12),
        ('VisualDynamics-0.0.1-windows-x64-setup.exe', 9)])
    names = [a['name'] for a in manifest['assets']]
    assert names == ['VisualDynamics-0.1.0-macos-arm64.dmg',
                     'VisualDynamics-0.1.0-windows-x64-setup.exe']
    assert manifest['tag_name'] == 'v0.1.0'
    # ten sorts after nine: versions compare as numbers, not text
    later = manifest_for([('VisualDynamics-0.9.0-macos-arm64.dmg', 1),
                          ('VisualDynamics-0.10.0-macos-arm64.dmg', 2)])
    assert [a['name'] for a in later['assets']] == [
        'VisualDynamics-0.10.0-macos-arm64.dmg']


def test_the_pypi_jobs_publish_by_identity_and_only_when_meant():
    """`pip install visualdynamics` on the downloads page is a promise
    the release workflow keeps: the real distribution goes up when the
    release is published, by trusted publishing (id-token, no secret),
    and a hand-run `reserve` holds the name with an empty placeholder
    — without starting the Linux and Windows builds (2026-09-03)."""
    import os

    import yaml

    root = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(root, '.github', 'workflows', 'release.yml'),
              encoding='utf-8') as handle:
        workflow = yaml.safe_load(handle)
    jobs = workflow['jobs']
    for name in ('pypi', 'reserve'):
        job = jobs[name]
        assert job['environment'] == 'pypi'
        assert job['permissions'] == {'id-token': 'write'}, 'trusted publishing'
        publish = [s for s in job['steps'] if 'pypi-publish' in s.get('uses', '')]
        assert len(publish) == 1
        assert not any('password' in s.get('with', {}) for s in job['steps'])
    assert jobs['pypi']['if'] == ("github.event_name == 'release' && "
                                  "startsWith(github.event.release.tag_name, 'v')"), \
        'PyPI publishes for a version release only (2026-09-14)'
    assert 'inputs.reserve' in jobs['reserve']['if']
    for name in ('linux', 'windows', 'macos'):
        assert '!inputs.reserve' in jobs[name]['if'], (
            f'{name} would build on a reserve run')
    assert workflow[True]['workflow_dispatch']['inputs']['reserve']['default'] is False


def test_the_release_publishes_checksums_for_every_asset():
    """The downloads page promises notes and checksums (2026-09-11): the
    publish job writes SHA256SUMS over every build before the draft
    opens — the macOS pair among them since it builds on the runners
    (2026-10-10) — and the desk's fallback script that attaches
    hand-built macOS images adds their lines to the same file."""
    import pathlib

    import yaml

    ROOT = pathlib.Path(__file__).resolve().parent.parent
    workflow = yaml.safe_load(
        (ROOT / '.github' / 'workflows' / 'release.yml').read_text(encoding='utf-8'))
    steps = workflow['jobs']['publish']['steps']
    checksums = [s for s in steps if s.get('name') == 'Checksums']
    assert len(checksums) == 1 and 'sha256sum * > SHA256SUMS' in checksums[0]['run']
    upload = next(i for i, s in enumerate(steps) if 'gh-release' in s.get('uses', ''))
    assert steps.index(checksums[0]) < upload, 'summed before the assets are attached'
    script = ROOT / 'packaging' / 'attach_macos.sh'
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding='utf-8')
    assert 'SHA256SUMS' in text and '--clobber' in text and 'stapler validate' in text
    page = (ROOT / 'web' / 'launch' / 'downloads.html').read_text(encoding='utf-8')
    assert 'shasum -a 256 -c SHA256SUMS' in page


def test_the_staging_parser_agrees_with_the_update_check():
    """`stage_downloads.version_of` is a deliberate second parser — the
    script runs under the system python and cannot import the package
    — so the two are held to one answer here, alpha and release alike."""
    from visualdynamics.update import parse_version

    for text in ('0.1.0', '0.1.0a1', '0.1.0a10', '0.2.0b3', '1.0.0rc1',
                 '0.10.2'):
        assert version_of(f'VisualDynamics-{text}-macos-arm64.dmg') == \
            parse_version(text), text


def test_ci_runs_on_every_push_to_main_and_both_pythons():
    """The repository is public and its runners are free (2026-09-14),
    so the suite runs on every push to main, on pull requests and on
    demand, on both Pythons the package promises — and weekly, because
    unpinned dependencies make an upstream release something new to
    test even when nothing here moved (2026-10-09; it was "no schedule"
    until PySide6 6.12 waited for a release to be found)."""
    import yaml

    root = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(root, '.github', 'workflows', 'ci.yml'),
              encoding='utf-8') as handle:
        workflow = yaml.safe_load(handle)
    on = workflow[True]
    assert on['push'] == {'branches': ['main']}
    assert 'pull_request' in on and 'workflow_dispatch' in on
    assert on['schedule'] == [{'cron': '0 13 * * 1'}], 'weekly'
    matrix = workflow['jobs']['test']['strategy']['matrix']
    assert matrix['python-version'] == ['3.12', '3.13']


def test_the_publish_job_waits_for_every_build():
    """`needs` names all three builds (2026-08-25: it did not wait for
    macOS), and since the macOS pair builds on the runners (2026-10-10)
    the gate requires all three to have succeeded: a draft without its
    Mac images is the half-release the job exists to prevent. It
    downloads the build artifacts by name — every artifact took the
    smoke test's empty stdout.log too, which GitHub refuses as an
    asset (the first tag, 2026-09-14)."""
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    workflow = yaml.safe_load(
        (root / '.github' / 'workflows' / 'release.yml').read_text(encoding='utf-8'))
    publish = workflow['jobs']['publish']
    assert publish['needs'] == ['linux', 'windows', 'macos']
    gate = ' '.join(publish['if'].split())
    assert gate.startswith('!cancelled()')
    assert "needs.linux.result == 'success'" in gate
    assert "needs.windows.result == 'success'" in gate
    assert "needs.macos.result == 'success'" in gate, 'the Mac pair is required'
    download = next(s for s in publish['steps']
                    if 'download-artifact' in s.get('uses', ''))
    assert download['with']['pattern'] == '{linux,windows,macos-*}'


def test_the_macos_pair_builds_signed_on_a_runner_of_each_architecture():
    """The macOS images build on GitHub's runners (2026-10-10), one per
    architecture on a runner of that architecture, signed from the
    certificate secret and notarized from the Apple ID secrets, and on
    a tag the build refuses to finish unsigned or unnotarized. Each
    uploads its image and its update zip under its own name."""
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    workflow = yaml.safe_load(
        (root / '.github' / 'workflows' / 'release.yml').read_text(encoding='utf-8'))
    job = workflow['jobs']['macos']
    assert '!inputs.reserve' in job['if'] and '!inputs.site' in job['if']
    assert "github.event_name != 'release'" in job['if']
    assert 'macos' not in workflow[True]['workflow_dispatch']['inputs'], (
        'no longer off by default')
    matrix = job['strategy']['matrix']['include']
    assert {entry['arch'] for entry in matrix} == {'arm64', 'x86_64'}
    assert any('intel' in entry['runner'] for entry in matrix
               if entry['arch'] == 'x86_64')
    steps = job['steps']
    signing = next(s for s in steps if 'ci_signing.sh' in s.get('run', ''))
    assert set(signing['env']) == {'MACOS_CERTIFICATE', 'MACOS_CERTIFICATE_PASSWORD'}
    build = next(s for s in steps if 'build_macos.sh' in s.get('run', ''))
    assert build['env']['PYTHON'] == 'python', 'the runner has no .venv'
    assert {'NOTARY_APPLE_ID', 'NOTARY_PASSWORD'} <= set(build['env'])
    assert "startsWith(github.ref, 'refs/tags/v')" in build['env']['VD_REQUIRE_NOTARIZED']
    assert steps.index(signing) < steps.index(build)
    sparkle = next(s for s in steps if 'fetch_sparkle.sh' in s.get('run', ''))
    assert 'SPARKLE=' in sparkle['run'] and steps.index(sparkle) < steps.index(build)
    upload = next(s for s in steps if 'upload-artifact' in s.get('uses', ''))
    assert upload['with']['name'] == 'macos-${{ matrix.arch }}'
    assert 'dist/*.dmg' in upload['with']['path'] and 'dist/*.zip' in upload['with']['path']


def test_the_updates_are_signed_on_a_runner_after_the_draft_opens():
    """Sparkle's feeds go onto the draft after `publish` has written it
    (its notes are what Sparkle shows), from the EdDSA key secret
    written to a file only the job can read and removed after, with
    the two releases before this one fetched for the deltas."""
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    workflow = yaml.safe_load(
        (root / '.github' / 'workflows' / 'release.yml').read_text(encoding='utf-8'))
    job = workflow['jobs']['updates']
    assert job['needs'] == 'publish' and job['if'] == "github.event_name == 'push'"
    assert job['runs-on'].startswith('macos')
    run = '\n'.join(s.get('run', '') for s in job['steps'])
    assert 'head -2' in run and '--exclude-drafts' in run
    sign = next(s for s in job['steps'] if 'release_updates.sh' in s.get('run', ''))
    assert set(sign['env']) == {'SPARKLE_ED_KEY'}
    assert 'umask 077' in sign['run'] and 'SPARKLE_ED_KEY_FILE=' in sign['run']
    assert sign['run'].rstrip().endswith('rm -f "$RUNNER_TEMP/ed.key"')


def test_ci_keeps_its_measured_worker_count_and_ceiling():
    """The suite runs four workers on the runner again (`-n auto`) —
    after the window leak that grew each to 2.7 GB was fixed — under
    the sixty-minute ceiling that measurement allows, and one Python's
    failure does not cancel the other's run."""
    import yaml

    root = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(root, '.github', 'workflows', 'ci.yml'),
              encoding='utf-8') as handle:
        workflow = yaml.safe_load(handle)
    job = workflow['jobs']['test']
    assert job['timeout-minutes'] == 60
    assert job['strategy']['fail-fast'] is False
    runs = [s['run'] for s in job['steps'] if 'pytest tests' in s.get('run', '')]
    assert runs and all('-n auto' in run for run in runs)


def test_ci_holds_coverage_to_a_floor():
    """A pull request that adds code without tests goes red (Brandon,
    2026-09-16). The floor sits under the measured number — 92% on the
    suite alone the day it was set — and only ever moves up."""
    import re

    import yaml

    root = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(root, '.github', 'workflows', 'ci.yml'),
              encoding='utf-8') as handle:
        workflow = yaml.safe_load(handle)
    steps = workflow['jobs']['test']['steps']
    coverage = next(s for s in steps if s.get('name') == 'Coverage')
    floor = re.search(r'--fail-under=(\d+)', coverage['run'])
    assert floor, 'no floor'
    assert 85 <= int(floor.group(1)) <= 92, floor.group(1)


def test_the_site_can_be_redeployed_on_demand_between_releases():
    """A change to the pages themselves — the example tiles' pictures
    were the first (2026-09-16) — must not wait for a release: the
    release workflow's `site` job also runs on dispatch with the `site`
    input, and then writes the manifest from whatever release is
    latest so the update check keeps its answer."""
    import yaml

    root = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(root, '.github', 'workflows', 'release.yml'),
              encoding='utf-8') as handle:
        workflow = yaml.safe_load(handle)
    inputs = workflow[True]['workflow_dispatch']['inputs']
    assert inputs['site']['type'] == 'boolean' and inputs['site']['default'] is False
    site = workflow['jobs']['site']
    gate = ' '.join(site['if'].split())
    assert "github.event_name == 'release'" in gate
    assert "github.event_name == 'workflow_dispatch' && inputs.site" in gate
    manifest = next(s for s in site['steps'] if 'manifest' in s.get('name', ''))
    assert 'gh release view' in manifest['run'], 'the latest release names the manifest'
    assert 'GH_TOKEN' in manifest['env']
    # and a site-only dispatch builds nothing: the first one rebuilt
    # Linux and Windows for a page change (2026-09-16)
    for job in ('linux', 'windows', 'macos'):
        assert '!inputs.site' in workflow['jobs'][job]['if'], job
