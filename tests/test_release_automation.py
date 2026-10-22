"""A sync that carries a new version releases itself.

When the tests pass on a push to the public main and main carries a
`__version__` with no release yet, the release workflow builds it,
drafts it and publishes it, then
PyPI and the site follow (packaging/release_decision.py). These pin the
decision for every way the workflow starts, the version bump the sync
tool's `--release` makes, and the wiring of the workflow itself, which
can only run on GitHub.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decision = _load('packaging/release_decision.py', 'release_decision')
alpha = _load('tools/next_alpha.py', 'next_alpha')


def _green_main(**overrides):
    said = {'run_conclusion': 'success', 'run_event': 'push', 'run_branch': 'main',
            'run_sha': 'abc123', 'version': '0.1.0a67', 'released': False}
    said.update(overrides)
    return decision.decide('workflow_run', **said)


def test_a_green_main_with_a_new_version_releases_it():
    assert _green_main() == {'build': 'true', 'release': 'true', 'auto': 'true',
                             'tag': 'v0.1.0a67', 'sha': 'abc123'}


@pytest.mark.parametrize('why', [
    {'released': True},                 # the version has a release or a tag already
    {'run_conclusion': 'failure'},      # the tests did not pass
    {'run_event': 'pull_request'},      # not a push
    {'run_event': 'schedule'},          # the weekly run
    {'run_branch': 'sync/20261021-1636'},
    {'version': ''},
])
def test_anything_else_from_the_tests_releases_nothing(why):
    said = _green_main(**why)
    assert said['build'] == said['release'] == said['auto'] == 'false'
    assert said['tag'] == ''


def test_a_pushed_tag_releases_that_tag_and_is_published_by_hand():
    said = decision.decide('push', ref_name='v0.1.0a67', sha='def456')
    assert said == {'build': 'true', 'release': 'true', 'auto': 'false',
                    'tag': 'v0.1.0a67', 'sha': 'def456'}


def test_a_manual_run_builds_as_a_test_unless_it_only_reserves_or_redeploys():
    assert decision.decide('workflow_dispatch', sha='s')['build'] == 'true'
    assert decision.decide('workflow_dispatch', sha='s')['release'] == 'false'
    assert decision.decide('workflow_dispatch', reserve=True)['build'] == 'false'
    assert decision.decide('workflow_dispatch', site=True)['build'] == 'false'


def test_a_published_release_builds_nothing():
    said = decision.decide('release', sha='s')
    assert said['build'] == said['release'] == 'false'


def test_the_version_is_read_from_the_package():
    text = (ROOT / 'src' / 'visualdynamics' / '__init__.py').read_text(encoding='utf-8')
    import visualdynamics

    assert decision.version_in(text) == visualdynamics.__version__
    with pytest.raises(ValueError):
        decision.version_in('x = 1\n')


def test_the_next_alpha_counts_up_and_refuses_anything_else():
    assert alpha.next_alpha('0.1.0a66') == '0.1.0a67'
    assert alpha.next_alpha('0.1.0a9') == '0.1.0a10', 'a number, not a digit'
    for version in ('0.1.0', '0.1.0b1', '0.1.0rc1'):
        with pytest.raises(ValueError, match='not an alpha'):
            alpha.next_alpha(version)
    text, new = alpha.bump("x = 1\n__version__ = '0.1.0a66'\ny = 2\n")
    assert new == '0.1.0a67'
    assert text == "x = 1\n__version__ = '0.1.0a67'\ny = 2\n", 'nothing else touched'


def _workflow(name):
    return yaml.safe_load((ROOT / '.github' / 'workflows' / name).read_text(encoding='utf-8'))


def test_the_release_workflow_runs_on_the_tests_passing_on_main():
    release = _workflow('release.yml')
    trigger = release[True]['workflow_run']     # PyYAML reads `on:` as True
    tests = _workflow('ci.yml')['name']
    assert trigger['workflows'] == [tests], 'named as the tests workflow is named'
    assert trigger['branches'] == ['main'] and trigger['types'] == ['completed']
    jobs = release['jobs']
    decide = jobs['which']['steps'][-1]['run']
    assert 'packaging/release_decision.py' in decide
    for build in ('linux', 'windows', 'macos'):
        assert jobs[build]['needs'] == 'which'
        assert jobs[build]['if'] == "needs.which.outputs.build == 'true'"
    built_from = {step.get('with', {}).get('ref') for job in ('linux', 'windows', 'macos')
                  for step in jobs[job]['steps'] if step.get('uses', '').startswith('actions/checkout')}
    assert built_from == {'${{ needs.which.outputs.sha }}'}, 'the commit the tests passed on'
    notarized = jobs['macos']['steps'][-2]['env']['VD_REQUIRE_NOTARIZED']
    assert "needs.which.outputs.release == 'true'" in notarized, 'a sync release is notarized too'


def test_a_sync_release_is_a_draft_until_its_feeds_are_on_it():
    jobs = _workflow('release.yml')['jobs']
    drafted = next(step for step in jobs['publish']['steps']
                   if step.get('uses', '').startswith('softprops/'))['with']
    assert drafted['draft'] is True
    assert drafted['tag_name'] == '${{ needs.which.outputs.tag }}'
    assert drafted['target_commitish'] == '${{ needs.which.outputs.sha }}'
    publish = jobs['publish_draft']
    assert publish['environment'] == 'release', (
        'the environment that lets only the protected main through, and where a '
        'reviewer added would hold it')
    assert publish['needs'] == ['which', 'updates'], 'after the update feeds are on the draft'
    assert publish['if'] == "needs.which.outputs.auto == 'true'"
    assert '--draft=false' in publish['steps'][-1]['run']
    for after in ('pypi', 'site'):
        assert 'publish_draft' in jobs[after]['needs']
        assert "needs.publish_draft.result == 'success'" in jobs[after]['if']
        assert jobs[after]['if'].startswith('!cancelled()'), (
            'a skipped publish_draft (a pushed tag) must not skip the release event path')


def test_the_sync_tool_asks_for_the_release():
    tool = (ROOT / 'tools' / 'sync_public.sh').read_text(encoding='utf-8')
    assert 'tools/next_alpha.py' in tool and '--release' in tool
    assert 'git status --porcelain' in tool, 'uncommitted work is refused'
