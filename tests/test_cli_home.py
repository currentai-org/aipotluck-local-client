"""aipotluck.installer.cli_home -- source-version detection and the CLI-home manifest.

Everything is rooted under tmp_path: no real checkout of this project, no real CLI home. The tests
that need git run against a throwaway repo instead and skip when git isn't installed at all.

The two functions with fail-open contracts (`read_build_info` -> "unknown", `read_installed` ->
None) are named for what they actually promise rather than for the happy path, because those are
the answers that decide whether a real install is kept or replaced: "unknown" must never look like
a version that beats what's installed, and None must mean "nothing worth keeping", not "trust
whatever the manifest says".
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aipotluck.installer import cli_home
from aipotluck.installer.cli_home import BuildInfo, InstalledCli
from aipotluck.installer.layout import Layout, get_layout

SHA_A = "a" * 40
SHA_B = "b" * 40

GIT = shutil.which("git")
requires_git = pytest.mark.skipif(GIT is None, reason="git is not installed")


# --------------------------------------------------------------------------- helpers


def _write_build_info(root: Path, text: str) -> None:
    path = root / cli_home.BUILD_INFO_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _layout(tmp_path: Path) -> Layout:
    root = tmp_path / "install"
    return Layout(install_root=root, config_dir=root / "config", log_dir=root / "logs", state_dir=root / "state")


def _git(path: Path, *args: str) -> str:
    result = subprocess.run([GIT, *args], cwd=path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_repo_with_a_commit(path: Path) -> Path:
    """A one-commit repo whose `_build_info.txt` is still the unsubstituted placeholder -- i.e. what
    a `git clone` of this project looks like (export-subst only runs for `git archive`)."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _write_build_info(path, "$Format:%H %ct$\n")
    _git(path, "add", "-A")
    _git(path, "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", "one")
    return path


def _installed(home: Path, *, sha: str = SHA_A, commit_time: int = 100) -> InstalledCli:
    src = home / f"src-{sha[:12]}"
    src.mkdir(parents=True, exist_ok=True)
    python = src / "python3"
    python.write_text("", encoding="utf-8")
    return InstalledCli(BuildInfo(sha, commit_time), src, python, "cli", "2026-01-01T00:00:00Z")


def _valid_manifest(src: Path, python: Path) -> dict:
    return {
        "schema": cli_home.SCHEMA,
        "sha": SHA_A,
        "commit_time": 100,
        "src": str(src),
        "python": str(python),
        "installed_by": "cli",
        "installed_at": "2026-01-01T00:00:00Z",
    }


# --------------------------------------------------------------------- read_build_info


class TestReadBuildInfo:
    def test_substituted_build_info_is_used_as_is(self, tmp_path):
        root = tmp_path / "src"
        _write_build_info(root, f"{SHA_A} 1712345678\n")

        build = cli_home.read_build_info(root)

        assert build == BuildInfo(SHA_A, 1712345678)
        assert build.short() == SHA_A[:12]

    @requires_git
    def test_substituted_build_info_wins_over_the_checkout_it_sits_in(self, tmp_path):
        # A staged copy carries its own resolved values; if it ever happens to be stored inside a
        # git checkout (a source build under ~/src, say) those values must still be the answer.
        root = _init_repo_with_a_commit(tmp_path / "src")
        _write_build_info(root, f"{SHA_B} 5\n")

        assert cli_home.read_build_info(root) == BuildInfo(SHA_B, 5)

    def test_missing_build_info_outside_a_git_checkout_reads_as_unknown_never_raises(self, tmp_path):
        root = tmp_path / "src"
        root.mkdir()

        assert cli_home.read_build_info(root) == BuildInfo("unknown", 0)
        assert cli_home.read_build_info(tmp_path / "does-not-exist") == BuildInfo("unknown", 0)

    @requires_git
    def test_unsubstituted_build_info_outside_a_git_checkout_reads_as_unknown_never_a_parent_repo(
        self, tmp_path
    ):
        # The whole point of the "only if this root has a .git" guard: a CLI home under a HOME that
        # happens to be a git repo (dotfiles) must not be stamped with the dotfiles repo's commit.
        home_repo = _init_repo_with_a_commit(tmp_path / "home")
        parent_sha = _git(home_repo, "rev-parse", "HEAD")
        assert parent_sha != "unknown"
        root = home_repo / ".local" / "share" / "aipotluck" / "cli" / "src-unknown"
        _write_build_info(root, "$Format:%H %ct$\n")

        assert cli_home.read_build_info(root) == BuildInfo("unknown", 0)

    @pytest.mark.parametrize(
        "text",
        [
            "$Format:%H %ct$",  # unsubstituted, no trailing newline
            "",  # empty file
            "   \n",
            f"{SHA_A}",  # no timestamp
            f"{SHA_A} not-a-time",
            f"{SHA_A} 1 2",  # too many tokens
            f"{SHA_A} -5",  # not a unix timestamp
            "not-a-sha 123",
            f"{SHA_B[:12]} 123",  # abbreviated sha: only a full 40-hex one identifies a commit
        ],
    )
    def test_malformed_build_info_outside_a_git_checkout_reads_as_unknown(self, tmp_path, text):
        root = tmp_path / "src"
        _write_build_info(root, text)

        assert cli_home.read_build_info(root) == BuildInfo("unknown", 0)

    @requires_git
    def test_git_checkout_without_a_substituted_file_falls_back_to_head(self, tmp_path):
        root = _init_repo_with_a_commit(tmp_path / "src")
        head_sha = _git(root, "rev-parse", "HEAD")
        head_time = int(_git(root, "log", "-1", "--format=%ct"))

        assert cli_home.read_build_info(root) == BuildInfo(head_sha, head_time)

    @requires_git
    def test_git_checkout_with_a_malformed_build_info_falls_back_to_head(self, tmp_path):
        root = _init_repo_with_a_commit(tmp_path / "src")
        _write_build_info(root, "garbage\n")
        head_sha = _git(root, "rev-parse", "HEAD")

        assert cli_home.read_build_info(root).sha == head_sha

    def test_broken_or_absent_git_reads_as_unknown_never_raises(self, tmp_path):
        # A `.git` that isn't a repo at all (an empty directory is what a half-finished copy leaves
        # behind), and git itself missing from PATH. Both are "unknown", never an exception.
        broken = tmp_path / "broken"
        (broken / ".git").mkdir(parents=True)

        assert cli_home.read_build_info(broken) == BuildInfo("unknown", 0)

        no_git = tmp_path / "no-git"
        (no_git / ".git").mkdir(parents=True)
        monkeypatched = cli_home.subprocess.run

        def _explode(*args, **kwargs):
            raise FileNotFoundError("git")

        cli_home.subprocess.run = _explode
        try:
            assert cli_home.read_build_info(no_git) == BuildInfo("unknown", 0)
        finally:
            cli_home.subprocess.run = monkeypatched


class TestIsNewer:
    @pytest.mark.parametrize(
        "candidate, installed, expected",
        [
            (BuildInfo(SHA_A, 200), BuildInfo(SHA_B, 100), True),  # newer time, different commit
            (BuildInfo(SHA_A, 100), BuildInfo(SHA_B, 100), False),  # same second: not an upgrade
            (BuildInfo(SHA_B, 99), BuildInfo(SHA_A, 100), False),  # older: never a downgrade
            (BuildInfo("unknown", 0), BuildInfo(SHA_A, 100), False),
            (BuildInfo("unknown", 0), BuildInfo("unknown", 0), False),  # nothing beats nothing
            (BuildInfo(SHA_A, 1), BuildInfo("unknown", 0), True),  # a real commit beats unknown
        ],
    )
    def test_only_a_strictly_greater_commit_time_is_newer(self, candidate, installed, expected):
        assert cli_home.is_newer(candidate, installed) is expected


# --------------------------------------------------------------------------- cli_home()


class TestCliHome:
    def test_cli_home_is_the_install_roots_cli_directory(self, tmp_path):
        lay = _layout(tmp_path)
        assert cli_home.cli_home(lay) == lay.install_root / "cli"

    def test_cli_home_lands_in_the_per_user_data_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

        home = cli_home.cli_home(get_layout("linux", system_scope=False))

        assert home == tmp_path / "data" / "aipotluck" / "cli"


# ----------------------------------------------------------------------- read_installed


class TestReadInstalled:
    def test_round_trips_what_write_installed_wrote(self, tmp_path):
        home = tmp_path / "cli"
        written = _installed(home)

        manifest = cli_home.write_installed(home, written)

        assert manifest == home / cli_home.MANIFEST_NAME
        assert cli_home.read_installed(home) == written

    def test_returns_none_when_nothing_is_installed(self, tmp_path):
        assert cli_home.read_installed(tmp_path / "empty-home") is None

    @pytest.mark.parametrize("text", ["", "not json at all", "{", "[]", '"a string"', "null", "3"])
    def test_returns_none_for_a_manifest_that_is_not_a_json_object(self, tmp_path, text):
        home = tmp_path / "cli"
        home.mkdir()
        (home / cli_home.MANIFEST_NAME).write_text(text, encoding="utf-8")

        assert cli_home.read_installed(home) is None

    @pytest.mark.parametrize(
        "case",
        [
            "schema-2",
            "schema-is-a-bool",
            "no-schema",
            "commit-time-is-a-string",
            "empty-sha",
            "no-installed-by",
            "no-installed-at",
            "src-directory-deleted",
            "python-file-deleted",
            "src-is-a-file",
        ],
    )
    def test_returns_none_for_a_manifest_it_cannot_trust(self, tmp_path, case):
        home = tmp_path / "cli"
        home.mkdir()
        src = home / "src-aaaaaaaaaaaa"
        src.mkdir()
        python = src / "python3"
        python.write_text("", encoding="utf-8")
        payload = _valid_manifest(src, python)

        if case == "schema-2":
            payload["schema"] = 2
        elif case == "schema-is-a-bool":
            payload["schema"] = True  # True == 1 in Python; must not pass for schema 1
        elif case == "no-schema":
            del payload["schema"]
        elif case == "commit-time-is-a-string":
            payload["commit_time"] = "100"
        elif case == "empty-sha":
            payload["sha"] = ""
        elif case == "no-installed-by":
            del payload["installed_by"]
        elif case == "no-installed-at":
            del payload["installed_at"]
        elif case == "src-directory-deleted":
            shutil.rmtree(src)
        elif case == "python-file-deleted":
            python.unlink()
        elif case == "src-is-a-file":
            shutil.rmtree(src)
            src.write_text("", encoding="utf-8")
        (home / cli_home.MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")

        assert cli_home.read_installed(home) is None


# ---------------------------------------------------------------------- write_installed


class TestWriteInstalled:
    def test_creates_the_cli_home_and_leaves_no_temp_file_behind(self, tmp_path):
        home = tmp_path / "not" / "yet" / "there"

        cli_home.write_installed(home, _installed(tmp_path / "staging"))

        assert sorted(p.name for p in home.iterdir()) == [cli_home.MANIFEST_NAME]

    def test_replaces_an_earlier_manifest_rather_than_appending(self, tmp_path):
        home = tmp_path / "cli"
        cli_home.write_installed(home, _installed(tmp_path / "staging-a", sha=SHA_A, commit_time=100))
        newer = _installed(tmp_path / "staging-b", sha=SHA_B, commit_time=200)

        cli_home.write_installed(home, newer)

        assert cli_home.read_installed(home) == newer
        assert sorted(p.name for p in home.iterdir()) == [cli_home.MANIFEST_NAME]

    def test_a_failed_replace_leaves_the_previous_manifest_intact(self, tmp_path, monkeypatch):
        # The reason for temp-file-plus-os.replace at all: a crash mid-write must not leave a
        # truncated manifest that the next install reads as "nothing installed".
        home = tmp_path / "cli"
        original = _installed(tmp_path / "staging-a", sha=SHA_A, commit_time=100)
        cli_home.write_installed(home, original)
        before = (home / cli_home.MANIFEST_NAME).read_text(encoding="utf-8")

        def _fail(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(cli_home.os, "replace", _fail)
        with pytest.raises(OSError):
            cli_home.write_installed(home, _installed(tmp_path / "staging-b", sha=SHA_B, commit_time=200))

        assert (home / cli_home.MANIFEST_NAME).read_text(encoding="utf-8") == before
        assert sorted(p.name for p in home.iterdir()) == [cli_home.MANIFEST_NAME]
        assert cli_home.read_installed(home) == original
