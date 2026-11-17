"""aipotluck.installer.cli_home -- source-version detection, the CLI-home manifest, staging, and
the adopt/prune decisions.

Everything is rooted under tmp_path: no real checkout of this project, no real CLI home. The tests
that need git run against a throwaway repo instead and skip when git isn't installed at all.

The two functions with fail-open contracts (`read_build_info` -> "unknown", `read_installed` ->
None) are named for what they actually promise rather than for the happy path, because those are
the answers that decide whether a real install is kept or replaced: "unknown" must never look like
a version that beats what's installed, and None must mean "nothing worth keeping", not "trust
whatever the manifest says". The staging tests check the same way that an interrupted copy is
never mistaken for a finished one, and the adopt tests compare a whole-tree snapshot before and
after the calls that promise to write nothing -- "adopted" and "kept-newer" are only worth anything
if the home really is untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
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


def _fake_source(tmp_path: Path, *, sha: str = SHA_A, commit_time: int = 100, name: str = "source") -> Path:
    """A throwaway tree shaped like this project: a package, a stamped build-info file, and one of
    every directory the copy must leave behind (`.git`, `vendor`, `tests`, `research`, caches)."""
    root = tmp_path / name
    package = root / "aipotluck"
    (package / "__pycache__").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__pycache__" / "stale.pyc").write_text("bytecode", encoding="utf-8")
    _write_build_info(root, f"{sha} {commit_time}\n")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    for excluded in ("vendor", "tests", "research", ".pytest_cache"):
        (root / excluded).mkdir()
        (root / excluded / "thing").write_text(excluded, encoding="utf-8")
    (root / "run_local.py").write_text("print('hi')\n", encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict:
    """Every file's sha256 and mtime plus every directory, keyed by relative path -- enough that
    any write, copy or timestamp touch anywhere under root changes the result."""
    snap: dict = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        stat = path.lstat()
        if path.is_dir() and not path.is_symlink():
            snap[rel] = f"dir {stat.st_mtime_ns}"
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            snap[rel] = f"{stat.st_mtime_ns} {digest}"
    snap["<root>"] = f"dir {root.lstat().st_mtime_ns}"
    return snap


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


# ------------------------------------------------------------------------------ stage_source


class TestStageSource:
    def test_stages_the_tree_into_src_sha12_and_stamps_the_resolved_build_info(self, tmp_path):
        # The source is a plain checkout, so its own _build_info.txt is the unsubstituted
        # placeholder; the staged copy must carry the resolved values instead, since the .git it
        # could otherwise ask is not copied.
        source = _fake_source(tmp_path)
        _write_build_info(source, "$Format:%H %ct$\n")
        home = tmp_path / "cli"

        dest = cli_home.stage_source(source, home, BuildInfo(SHA_A, 1712345678))

        assert dest == home / f"src-{SHA_A[:12]}"
        assert dest.is_dir()
        assert (dest / cli_home.READY_MARKER).exists()
        assert (dest / cli_home.BUILD_INFO_RELPATH).read_text(encoding="utf-8") == f"{SHA_A} 1712345678\n"
        assert cli_home.read_build_info(dest) == BuildInfo(SHA_A, 1712345678)
        assert (dest / "run_local.py").read_text(encoding="utf-8") == "print('hi')\n"

    def test_reuses_a_copy_that_already_carries_the_ready_marker(self, tmp_path):
        # Reinstalling the same commit must cost nothing: a second stage of a complete copy leaves
        # every file and directory stamp alone, and keeps whatever is inside it.
        source = _fake_source(tmp_path)
        home = tmp_path / "cli"
        dest = cli_home.stage_source(source, home, BuildInfo(SHA_A, 100))
        (dest / "sentinel").write_text("still here", encoding="utf-8")
        before = _snapshot(dest)

        again = cli_home.stage_source(source, home, BuildInfo(SHA_A, 100))

        assert again == dest
        assert _snapshot(dest) == before
        assert (dest / "sentinel").read_text(encoding="utf-8") == "still here"
        assert sorted(p.name for p in home.iterdir()) == [dest.name]

    def test_an_unknown_sha_stages_into_src_unknown_never_a_partial_name(self, tmp_path):
        source = _fake_source(tmp_path, sha="unknown", commit_time=0)
        home = tmp_path / "cli"

        dest = cli_home.stage_source(source, home, BuildInfo("unknown", 0))

        assert dest == home / "src-unknown"
        assert (dest / cli_home.READY_MARKER).exists()
        assert (dest / cli_home.BUILD_INFO_RELPATH).read_text(encoding="utf-8") == "unknown 0\n"

    def test_excludes_git_vendor_tests_research_and_caches_from_the_copy(self, tmp_path):
        source = _fake_source(tmp_path)
        home = tmp_path / "cli"

        dest = cli_home.stage_source(source, home, BuildInfo(SHA_A, 100))

        copied = sorted(p.name for p in dest.iterdir())
        assert copied == sorted(["aipotluck", "run_local.py", cli_home.READY_MARKER])
        for excluded in (".git", "vendor", "tests", "research", ".pytest_cache"):
            assert not (dest / excluded).exists(), excluded
        assert not (dest / "aipotluck" / "__pycache__").exists()
        assert not list(dest.rglob("*.pyc"))
        assert (dest / "aipotluck" / "__init__.py").is_file()

    def test_replaces_a_stale_destination_without_a_ready_marker(self, tmp_path):
        # A half-copied tree under the final name (an installer killed before this module always
        # used the partial name) must be replaced, not trusted and not left in the way.
        source = _fake_source(tmp_path)
        home = tmp_path / "cli"
        stale = home / f"src-{SHA_A[:12]}"
        stale.mkdir(parents=True)
        (stale / "leftover").write_text("from an older crash", encoding="utf-8")

        dest = cli_home.stage_source(source, home, BuildInfo(SHA_A, 100))

        assert dest == stale
        assert (dest / cli_home.READY_MARKER).exists()
        assert not (dest / "leftover").exists()
        assert (dest / "run_local.py").is_file()

    def test_a_failed_copy_leaves_no_partial_directory_behind(self, tmp_path, monkeypatch):
        source = _fake_source(tmp_path)
        home = tmp_path / "cli"

        def _explode(src, dst, **kwargs):
            Path(dst).mkdir(parents=True)
            raise OSError("disk full")

        monkeypatch.setattr(cli_home.shutil, "copytree", _explode)
        with pytest.raises(OSError):
            cli_home.stage_source(source, home, BuildInfo(SHA_A, 100))

        assert sorted(p.name for p in home.iterdir()) == []
        assert not (home / f"src-{SHA_A[:12]}").exists()


# ------------------------------------------------------------------------------ stage_python


class TestStagePython:
    def test_copies_the_bundled_interpreter_verbatim_and_reuses_it(self, tmp_path):
        # No ignore list here on purpose: an interpreter's own layout is not ours to second-guess,
        # so even a directory this module would skip in the source tree is copied.
        bundled = tmp_path / "bundle"
        (bundled / "bin").mkdir(parents=True)
        (bundled / "bin" / "python3").write_text("#!/bin/sh\n", encoding="utf-8")
        (bundled / "lib").mkdir()
        (bundled / "lib" / "libpython3.12.so").write_text("elf", encoding="utf-8")
        (bundled / "tests").mkdir()
        (bundled / "tests" / "keep.txt").write_text("copied", encoding="utf-8")
        home = tmp_path / "cli"

        dest = cli_home.stage_python(bundled, home)

        assert dest == home / "python-bundle"
        assert (dest / "bin" / "python3").read_text(encoding="utf-8") == "#!/bin/sh\n"
        assert (dest / "lib" / "libpython3.12.so").is_file()
        assert (dest / "tests" / "keep.txt").is_file()
        assert (dest / cli_home.READY_MARKER).exists()

        (dest / "sentinel").write_text("still here", encoding="utf-8")
        assert cli_home.stage_python(bundled, home) == dest
        assert (dest / "sentinel").is_file()
        assert sorted(p.name for p in home.iterdir()) == [dest.name]


# ----------------------------------------------------------------------------- resolve_python


class TestResolvePython:
    def test_a_bundled_interpreter_is_staged_and_the_path_rebased_into_the_cli_home(self, tmp_path, monkeypatch):
        bundled = tmp_path / "bundle"
        (bundled / "bin").mkdir(parents=True)
        (bundled / "bin" / "python3").write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setattr(cli_home.sys, "executable", str(bundled / "bin" / "python3"))
        home = tmp_path / "cli"

        resolved = cli_home.resolve_python(bundled, home)

        assert resolved == home / "python-bundle" / "bin" / "python3"
        assert resolved.is_file()

    def test_an_executable_outside_the_bundled_dir_is_used_as_is_never_staged(self, tmp_path, monkeypatch):
        # install.sh runs under the system python3: copying that would duplicate the OS's
        # interpreter into the CLI home for no gain, so sys.executable is kept.
        bundled = tmp_path / "bundle"
        (bundled / "bin").mkdir(parents=True)
        system_python = tmp_path / "usr" / "bin" / "python3"
        system_python.parent.mkdir(parents=True)
        system_python.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setattr(cli_home.sys, "executable", str(system_python))
        home = tmp_path / "cli"

        assert cli_home.resolve_python(bundled, home) == system_python
        assert cli_home.resolve_python(None, home) == system_python
        assert not home.exists() or not any(p.name.startswith("python-") for p in home.iterdir())


# ------------------------------------------------------------------------------------- adopt


class TestAdopt:
    def test_fresh_adopt_stages_and_writes_a_manifest_pointing_into_the_cli_home(self, tmp_path):
        source = _fake_source(tmp_path)
        lay = _layout(tmp_path)
        home = cli_home.cli_home(lay)

        result = cli_home.adopt(source, lay)

        assert result.action == "adopted"
        assert result.installed.build == BuildInfo(SHA_A, 100)
        assert result.installed.src == home / f"src-{SHA_A[:12]}"
        assert result.installed.src.is_dir()
        assert (result.installed.src / cli_home.READY_MARKER).exists()
        assert result.installed.python == Path(sys.executable).absolute()
        assert result.installed.installed_by == "cli"
        assert result.installed.installed_at.endswith("Z")
        assert cli_home.read_installed(home) == result.installed
        assert sorted(p.name for p in home.iterdir()) == sorted(
            [cli_home.MANIFEST_NAME, f"src-{SHA_A[:12]}"]
        )

    def test_adopting_the_same_sha_again_is_current_and_writes_nothing(self, tmp_path):
        source = _fake_source(tmp_path)
        lay = _layout(tmp_path)
        home = cli_home.cli_home(lay)
        first = cli_home.adopt(source, lay)
        before = _snapshot(home)

        again = cli_home.adopt(source, lay)

        assert again.action == "current"
        assert again.installed == first.installed
        assert _snapshot(home) == before

    def test_an_older_candidate_is_kept_newer_and_leaves_the_home_byte_identical(self, tmp_path):
        # The whole point of the shared CLI home: the GUI and install.sh can run in either order,
        # and the older of the two must never win.
        newer = _fake_source(tmp_path, sha=SHA_B, commit_time=200, name="newer")
        older = _fake_source(tmp_path, sha=SHA_A, commit_time=100, name="older")
        lay = _layout(tmp_path)
        home = cli_home.cli_home(lay)
        installed = cli_home.adopt(newer, lay)
        before = _snapshot(home)

        result = cli_home.adopt(older, lay)

        assert result.action == "kept-newer"
        assert result.installed == installed.installed
        assert _snapshot(home) == before
        assert not (home / f"src-{SHA_A[:12]}").exists()

    def test_an_unknown_build_never_replaces_a_real_install(self, tmp_path):
        # A copy of a copy has no commit to compare; it must lose to anything already installed
        # rather than be taken as an upgrade over it.
        real = _fake_source(tmp_path, sha=SHA_A, commit_time=100, name="real")
        unknown = tmp_path / "unknown"
        (unknown / "aipotluck").mkdir(parents=True)
        (unknown / "run_local.py").write_text("print('hi')\n", encoding="utf-8")
        lay = _layout(tmp_path)
        home = cli_home.cli_home(lay)
        installed = cli_home.adopt(real, lay)
        before = _snapshot(home)

        result = cli_home.adopt(unknown, lay)

        assert result.action == "kept-newer"
        assert result.installed == installed.installed
        assert _snapshot(home) == before
        assert not (home / "src-unknown").exists()

    def test_force_reinstalls_an_older_candidate(self, tmp_path):
        newer = _fake_source(tmp_path, sha=SHA_B, commit_time=200, name="newer")
        older = _fake_source(tmp_path, sha=SHA_A, commit_time=100, name="older")
        lay = _layout(tmp_path)
        home = cli_home.cli_home(lay)
        cli_home.adopt(newer, lay)

        result = cli_home.adopt(older, lay, force=True, installed_by="gui")

        assert result.action == "adopted"
        assert result.installed.build == BuildInfo(SHA_A, 100)
        assert result.installed.src == home / f"src-{SHA_A[:12]}"
        assert result.installed.installed_by == "gui"
        assert cli_home.read_installed(home) == result.installed

    def test_a_corrupt_manifest_is_adopted_over_rather_than_trusted(self, tmp_path):
        source = _fake_source(tmp_path)
        lay = _layout(tmp_path)
        home = cli_home.cli_home(lay)
        home.mkdir(parents=True)
        (home / cli_home.MANIFEST_NAME).write_text("{not json", encoding="utf-8")

        result = cli_home.adopt(source, lay)

        assert result.action == "adopted"
        assert cli_home.read_installed(home) == result.installed

    def test_a_bundled_python_is_staged_and_recorded_in_the_manifest(self, tmp_path, monkeypatch):
        source = _fake_source(tmp_path)
        bundled = tmp_path / "bundle"
        (bundled / "bin").mkdir(parents=True)
        (bundled / "bin" / "python3").write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setattr(cli_home.sys, "executable", str(bundled / "bin" / "python3"))
        lay = _layout(tmp_path)
        home = cli_home.cli_home(lay)

        result = cli_home.adopt(source, lay, bundled_python_dir=bundled, installed_by="gui")

        assert result.action == "adopted"
        assert result.installed.python == home / "python-bundle" / "bin" / "python3"
        assert result.installed.python.is_file()
        assert cli_home.read_installed(home) == result.installed


# ------------------------------------------------------------------------------------- prune


class TestPrune:
    def test_removes_partials_and_superseded_copies_but_keeps_the_current_one(self, tmp_path):
        home = tmp_path / "cli"
        current = home / f"src-{SHA_A[:12]}"
        current.mkdir(parents=True)
        superseded = home / f"src-{SHA_B[:12]}"
        superseded.mkdir()
        partial_src = home / f".src-{SHA_B[:12]}.partial-123"
        partial_src.mkdir()
        partial_python = home / ".python-bundle.partial-123"
        partial_python.mkdir()

        removed = cli_home.prune(home, [current])

        assert sorted(p.name for p in removed) == sorted(
            [superseded.name, partial_src.name, partial_python.name]
        )
        assert current.is_dir()
        assert not superseded.exists()
        assert not partial_src.exists()
        assert not partial_python.exists()

    def test_never_removes_the_tree_holding_the_running_interpreter(self, tmp_path, monkeypatch):
        # Windows holds the interpreter's own files open: a tree the running process is executing
        # out of must survive even when the caller does not list it as the current copy.
        home = tmp_path / "cli"
        live = home / "src-live"
        (live / "bin").mkdir(parents=True)
        executable = live / "bin" / "python3"
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        doomed = home / "src-doomed"
        doomed.mkdir()
        monkeypatch.setattr(cli_home.sys, "executable", str(executable))

        removed = cli_home.prune(home, [])

        assert [p.name for p in removed] == [doomed.name]
        assert live.is_dir()
        assert executable.is_file()

    def test_never_touches_the_manifest_or_unrelated_files(self, tmp_path):
        # Anyone can put anything in the install root; prune only owns staged copies.
        home = tmp_path / "cli"
        home.mkdir(parents=True)
        (home / cli_home.MANIFEST_NAME).write_text("{}", encoding="utf-8")
        (home / "notes.txt").write_text("mine", encoding="utf-8")
        (home / "models").mkdir()
        (home / "models" / "m.gguf").write_text("weights", encoding="utf-8")
        (home / "src-stale").mkdir()

        removed = cli_home.prune(home, [])

        assert [p.name for p in removed] == ["src-stale"]
        assert (home / cli_home.MANIFEST_NAME).is_file()
        assert (home / "notes.txt").read_text(encoding="utf-8") == "mine"
        assert (home / "models" / "m.gguf").is_file()

    def test_a_directory_that_cannot_be_removed_is_a_warning_and_the_rest_continue(
        self, tmp_path, monkeypatch, caplog
    ):
        home = tmp_path / "cli"
        stuck = home / "src-stuck"
        stale = home / "src-stale"
        stuck.mkdir(parents=True)
        stale.mkdir()
        real_rmtree = shutil.rmtree

        def _flaky(path, *args, **kwargs):
            if Path(path).name == "src-stuck":
                raise OSError("held open by another process")
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(cli_home.shutil, "rmtree", _flaky)
        with caplog.at_level(logging.WARNING, logger="aipotluck.installer.cli_home"):
            removed = cli_home.prune(home, [])

        assert [p.name for p in removed] == ["src-stale"]
        assert stuck.is_dir()
        assert not stale.exists()
        assert any("src-stuck" in record.getMessage() for record in caplog.records)

    def test_a_missing_cli_home_removes_nothing_never_raises(self, tmp_path):
        assert cli_home.prune(tmp_path / "not-there", []) == []
