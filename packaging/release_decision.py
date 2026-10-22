"""What one run of the release workflow does: build or not, release or
not, and which tag and commit.

    python packaging/release_decision.py >> "$GITHUB_OUTPUT"

The workflow starts four ways. A pushed `v…` tag releases that tag, as
it always has. A manual run builds all four packages as a test and
releases nothing, unless it was asked only to reserve the PyPI name or
redeploy the site. A published release builds nothing; its PyPI and
site jobs run from the release itself. And when the tests pass on a
push to `main`, the run releases `main` if the version it carries has
no release yet. A sync with a new `__version__` is the request to
release; a sync that leaves the version alone releases nothing.

That last path ends at the `approve` job, which waits for the
maintainer's approval before the draft is published, and only then do
PyPI and the site follow. GitHub does not start a workflow from a tag
or a release that a workflow's own token made, which is why one run
carries the release from the tests' result to the website.

Everything comes from the environment the workflow sets. The decision
itself is `decide`, which takes plain values and touches nothing, so
it can be tested without GitHub.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / 'src' / 'visualdynamics' / '__init__.py'


def version_in(text: str) -> str:
    """The `__version__` assigned in a module's source."""
    found = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    if found is None:
        raise ValueError('no __version__ in the package')
    return found.group(1)


def decide(event: str, *, ref_name: str = '', sha: str = '', reserve: bool = False,
           site: bool = False, run_conclusion: str = '', run_event: str = '',
           run_branch: str = '', run_sha: str = '', version: str = '',
           released: bool = False) -> dict[str, str]:
    """What this run does, as the workflow's outputs: `build`,
    `release` and `auto` ('true' or 'false'), and the `tag` and `sha`
    to release and build.

    `released` says whether a release (draft or published) or a tag
    already has the name `v<version>`.
    """
    nothing = {'build': 'false', 'release': 'false', 'auto': 'false', 'tag': '', 'sha': sha}
    if event == 'push' and ref_name.startswith('v'):
        return {**nothing, 'build': 'true', 'release': 'true', 'tag': ref_name}
    if event == 'workflow_dispatch':
        return {**nothing, 'build': 'false' if reserve or site else 'true'}
    if event == 'workflow_run':
        if (run_conclusion != 'success' or run_event != 'push' or run_branch != 'main'
                or not version or released):
            return {**nothing, 'sha': run_sha}
        return {'build': 'true', 'release': 'true', 'auto': 'true',
                'tag': f'v{version}', 'sha': run_sha}
    return nothing


def released(tag: str, repository: str) -> bool:
    """Whether GitHub already has a release, draft included, or a tag
    named `tag`. A release that was drafted and never approved still
    holds its name, so a rerun does not draft it twice."""
    for command in (['gh', 'release', 'view', tag, '--repo', repository],
                    ['gh', 'api', f'repos/{repository}/git/ref/tags/{tag}']):
        if subprocess.run(command, check=False, capture_output=True).returncode == 0:
            return True
    return False


def main() -> int:
    env = os.environ
    event = env.get('EVENT', '')
    version = ''
    already = False
    if event == 'workflow_run':
        version = version_in(VERSION_FILE.read_text(encoding='utf-8'))
        already = released(f'v{version}', env['GITHUB_REPOSITORY'])
    outputs = decide(event, ref_name=env.get('GITHUB_REF_NAME', ''),
                     sha=env.get('GITHUB_SHA', ''),
                     reserve=env.get('RESERVE') == 'true', site=env.get('SITE') == 'true',
                     run_conclusion=env.get('RUN_CONCLUSION', ''),
                     run_event=env.get('RUN_EVENT', ''),
                     run_branch=env.get('RUN_BRANCH', ''), run_sha=env.get('RUN_SHA', ''),
                     version=version, released=already)
    for key, value in outputs.items():
        print(f'{key}={value}')
    if event == 'workflow_run':
        said = (f'releasing v{version}' if outputs['release'] == 'true'
                else f'v{version} has a release already, or the tests did not pass on a '
                     'push to main: nothing to release')
        print(said, file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
