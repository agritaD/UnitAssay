"""code_commit is '+dirty' only when the PACKAGE changed, never for other files.

2026-10-01: editing paper/ during a run stamped 141 of 147 sessions '+dirty'
with zero package changes. Built on a throwaway repository so the test does not
depend on the state of this checkout.
"""
import subprocess

import pytest

from unitassay import code_commit


def _git(repo, *a):
    subprocess.run(['git', '-C', str(repo), *a], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / 'unitassay').mkdir()
    (tmp_path / 'paper').mkdir()
    (tmp_path / 'unitassay' / 'core.py').write_text('x = 1\n')
    (tmp_path / 'paper' / 'fig.py').write_text('y = 1\n')
    (tmp_path / '.gitignore').write_text('__pycache__/\n')
    _git(tmp_path, 'init', '-q')
    _git(tmp_path, '-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '-A')
    _git(tmp_path, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'init')
    return tmp_path


def test_clean_checkout_is_not_dirty(repo):
    c = code_commit(repo)
    assert c and not c.endswith('+dirty')


def test_a_change_outside_the_package_is_not_dirty(repo):
    (repo / 'paper' / 'fig.py').write_text('y = 2\n')
    (repo / 'paper' / 'new.py').write_text('z = 1\n')
    assert not code_commit(repo).endswith('+dirty')


def test_a_modified_package_file_is_dirty(repo):
    (repo / 'unitassay' / 'core.py').write_text('x = 2\n')
    assert code_commit(repo).endswith('+dirty')


def test_an_untracked_package_file_is_dirty(repo):
    (repo / 'unitassay' / 'new_rule.py').write_text('r = 1\n')
    assert code_commit(repo).endswith('+dirty')


def test_ignored_files_in_the_package_are_not_dirty(repo):
    (repo / 'unitassay' / '__pycache__').mkdir()
    (repo / 'unitassay' / '__pycache__' / 'core.pyc').write_bytes(b'\0')
    assert not code_commit(repo).endswith('+dirty')


def test_outside_a_checkout_is_none(tmp_path):
    assert code_commit(tmp_path) is None
