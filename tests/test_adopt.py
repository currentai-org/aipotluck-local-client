"""aipotluck.installer.adopt -- the entry point both installers' CLI-home flow goes through, and
the `ADOPT_RESULT <json>` line the GUI parses.

What is asserted here is adopt's own wiring: when it does and doesn't touch the OS service, that
the service and the shim are pointed at the *staged* copy rather than the tree that ran the
install, and that every outcome -- adopted, current, kept-newer, skipped, error -- ends with a
parseable last stdout line. The staging, manifest and version decisions themselves are
cli_home.py's (tests/test_cli_home.py); the shim text and PATH handling are cli_shim.py's
(tests/test_cli_shim.py). The fake service manager is a MagicMock, since registering a real unit is
exactly what these tests must never do.

Everything is rooted under tmp_path: a fake HOME for the shell-startup files, a fake bin dir for
the shim, a fake source tree with a chosen build-info stamp, and a Layout under tmp_path -- no real
units, no real PATH, no real checkout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aipotluck.installer import adopt, cli_home, cli_shim
from aipotluck.installer.cli_home import BuildInfo, InstalledCli
from aipotluck.installer.layout import Layout
from aipotluck.installer.platform_detect import HostProfile
from aipotluck.installer.service.base import ServiceState, ServiceStatus

SHA_OLD = "a" * 40
SHA_NEW = "b" * 40
TIME_OLD = 1_000_000
TIME_NEW = 1_100_000

# What adopt builds from detect_os()/detect_arch() below -- only os_name reaches a code path here
# (the layout, the service manager and the shim's bin dir).
PROFILE = HostProfile(os_name="linux", arch="arm64", backend="cpu")


# --------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def _sandboxed_host(monkeypatch, tmp_path):
    """Neither the real ~/.local/bin nor the real ~/.profile may be reached: the shim goes to a tmp
    bin dir and `add_to_shell_profiles`' default `Path.home()` is redirected at a tmp HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cli_shim, "_bin_dir", lambda profile, system_scope: tmp_path / "bin")
    monkeypatch.setattr(cli_shim, "_login_shell", lambda: "bash")
    monkeypatch.setattr(cli_shim.Path, "home", classmethod(lambda cls: home))


def _layout(tmp_path: Path) -> Layout:
    root = tmp_path / "install"
    return Layout(install_root=root, config_dir=root / "config", log_dir=root / "logs", state_dir=root / "state")


def _fake_source(tmp_path: Path, *, sha: str, commit_time: int, name: str = "source") -> Path:
    """A throwaway tree shaped like this project: the package with a stamped build-info file (so its
    version is the one the test chose), and its own installer directory."""
    root = tmp_path / name
    package = root / "aipotluck"
    (package / "installer").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "installer" / "cli.py").write_text("# cli\n", encoding="utf-8")
    (package / "_build_info.txt").write_text(f"{sha} {commit_time}\n", encoding="utf-8")
    return root


def _fake_manager(
    monkeypatch, *, state: ServiceState = ServiceState.RUNNING, install_error=None, start_error=None
):
    """A MagicMock service manager plus the factory that hands it out -- asserting on the factory
    proves adopt never even looked at a service manager it had no reason to touch.

    Patched on cli_home (where it is imported at module level), which is also the seam adopt itself
    goes through, so a single patch reaches both the status check and the registration."""
    mgr = MagicMock()
    mgr.status.return_value = ServiceStatus(state, detail="fake")
    if install_error is not None:
        mgr.install.side_effect = install_error
    if start_error is not None:
        mgr.start.side_effect = start_error
    factory = MagicMock(return_value=mgr)
    monkeypatch.setattr(cli_home, "get_service_manager", factory)
    return factory, mgr


def _run(tmp_path, monkeypatch, *, source: Path, argv: list[str], lay: Layout | None = None):
    lay = lay or _layout(tmp_path)
    monkeypatch.setattr(adopt.layout, "get_layout", lambda os_name, **kwargs: lay)
    monkeypatch.setattr(adopt, "detect_os", lambda: "linux")
    monkeypatch.setattr(adopt, "detect_arch", lambda: "arm64")
    monkeypatch.setattr(adopt, "REPO_ROOT", source)
    return adopt.main(argv), lay


def _result(capsys) -> dict:
    """The last stdout line, parsed the way the GUI parses it."""
    lines = capsys.readouterr().out.splitlines()
    assert lines, "adopt printed nothing to stdout"
    assert lines[-1].startswith(adopt.RESULT_PREFIX), lines[-1]
    return json.loads(lines[-1][len(adopt.RESULT_PREFIX):])


def _installed(lay: Layout) -> InstalledCli:
    inst = cli_home.read_installed(cli_home.cli_home(lay))
    assert inst is not None
    return inst


def _shim_text(tmp_path: Path) -> str:
    return (tmp_path / "bin" / cli_shim.CLI_SHIM_NAME).read_text(encoding="utf-8")


def _plant_manifest(lay: Layout, *, sha: str, commit_time: int) -> InstalledCli:
    """An already-installed CLI home, exactly as a previous adopt would have left it: the staged tree,
    the manifest -- and the shim on PATH pointing at that copy, which is what makes a repeat run a
    no-op rather than a re-point."""
    home = cli_home.cli_home(lay)
    src = home / f"src-{sha[:12]}"
    (src / "aipotluck" / "installer").mkdir(parents=True)
    (src / "aipotluck" / "installer" / "cli.py").write_text("# cli\n", encoding="utf-8")
    python = src / "python3"
    python.write_text("", encoding="utf-8")
    inst = InstalledCli(BuildInfo(sha, commit_time), src, python, "cli", "2026-01-01T00:00:00Z")
    cli_home.write_installed(home, inst)
    cli_shim.install_cli_shim(PROFILE, python, False, cli_root=src)
    return inst


# ----------------------------------------------------------------- the output contract


class TestResultLine:
    def test_the_last_stdout_line_is_the_result_json(self, tmp_path, monkeypatch, capsys):
        _fake_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        rc, lay = _run(tmp_path, monkeypatch, source=source, argv=[])
        result = _result(capsys)

        assert rc == 0
        inst = _installed(lay)
        assert result == {
            "action": "adopted",
            "sha": SHA_NEW,
            "installed_sha": SHA_NEW,
            "repointed": True,
            "service_restarted": False,
            "python": str(inst.python),
            "src": str(inst.src),
        }

    def test_a_failure_prints_an_error_result_line_and_exits_1(self, tmp_path, monkeypatch, capsys):
        # The GUI parses this line; a crash that printed only a traceback would look like "no
        # answer" rather than "failed", and the launch would have nothing to report.
        _fake_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        def _boom(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(cli_home, "stage_source", _boom)
        rc, _ = _run(tmp_path, monkeypatch, source=source, argv=[])

        assert rc == 1
        assert _result(capsys) == {"action": "error", "error": "no space left on device"}

    def test_installed_by_is_recorded_in_the_manifest(self, tmp_path, monkeypatch, capsys):
        _fake_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        _run(tmp_path, monkeypatch, source=source, argv=["--installed-by", "gui"])

        assert _installed(_layout(tmp_path)).installed_by == "gui"


# ------------------------------------------------------------------------ --if-installed


class TestIfInstalled:
    def test_skips_and_writes_nothing_when_runtime_json_is_missing(self, tmp_path, monkeypatch, capsys):
        """The GUI runs adopt on every launch; on a machine that has never installed anything it
        must leave no trace at all -- no CLI home, no shim, no service manager touched."""
        factory, mgr = _fake_manager(monkeypatch)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        rc, lay = _run(tmp_path, monkeypatch, source=source, argv=["--if-installed"])

        assert rc == 0
        assert _result(capsys) == {"action": "skipped", "reason": "not installed"}
        assert not (lay.install_root / cli_home.CLI_HOME_DIRNAME).exists()
        assert not (tmp_path / "bin").exists()
        factory.assert_not_called()
        mgr.install.assert_not_called()

    def test_adopts_when_runtime_json_exists(self, tmp_path, monkeypatch, capsys):
        _fake_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)
        lay = _layout(tmp_path)
        lay.config_dir.mkdir(parents=True)
        (lay.config_dir / "runtime.json").write_text("{}", encoding="utf-8")
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        rc, _ = _run(tmp_path, monkeypatch, source=source, argv=["--if-installed"], lay=lay)

        assert rc == 0
        assert _result(capsys)["action"] == "adopted"
        assert (lay.install_root / cli_home.CLI_HOME_DIRNAME / cli_home.MANIFEST_NAME).exists()


# --------------------------------------------------------------------- a fresh adopt


class TestFreshAdopt:
    def test_a_running_service_is_stopped_repointed_and_restarted(self, tmp_path, monkeypatch, capsys):
        _, mgr = _fake_manager(monkeypatch, state=ServiceState.RUNNING)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        rc, lay = _run(tmp_path, monkeypatch, source=source, argv=[])
        result = _result(capsys)

        assert rc == 0
        assert result["action"] == "adopted"
        assert result["service_restarted"] is True
        mgr.stop.assert_called_once_with(cli_home.SERVICE_NAME)

        inst = _installed(lay)
        args, kwargs = mgr.install.call_args
        assert args[0] == cli_home.SERVICE_NAME
        assert kwargs["exec_path"] == inst.python
        assert kwargs["args"] == [
            str(inst.src / "aipotluck" / "service" / "aipotluck_service.py"),
            "--config-dir", str(lay.config_dir),
            "--log-dir", str(lay.log_dir),
        ]
        assert kwargs["working_dir"] == inst.src
        assert kwargs["log_dir"] == lay.log_dir
        assert kwargs["system_scope"] is False
        mgr.start.assert_called_once_with(cli_home.SERVICE_NAME, system_scope=False)

    def test_the_service_script_and_working_dir_are_under_the_cli_home_not_the_source_tree(
        self, tmp_path, monkeypatch, capsys
    ):
        # The whole point: a GUI bundled with this tree may be uninstalled or replaced tomorrow; the
        # service must keep running the copy staged into the CLI home.
        _, mgr = _fake_manager(monkeypatch, state=ServiceState.RUNNING)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        _run(tmp_path, monkeypatch, source=source, argv=[])
        _result(capsys)

        _, kwargs = mgr.install.call_args
        cli_root = _layout(tmp_path).install_root / cli_home.CLI_HOME_DIRNAME
        assert cli_root in Path(kwargs["args"][0]).parents
        assert cli_root in Path(kwargs["working_dir"]).parents
        assert str(source) not in str(kwargs["args"][0])

    def test_the_shim_points_into_the_cli_home(self, tmp_path, monkeypatch, capsys):
        _fake_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        _run(tmp_path, monkeypatch, source=source, argv=[])
        _result(capsys)

        inst = _installed(_layout(tmp_path))
        text = _shim_text(tmp_path)
        assert f'exec "{inst.python}" "{inst.src / "aipotluck" / "installer" / "cli.py"}" "$@"' in text
        assert str(source) not in text

    def test_older_staged_copies_are_pruned_but_the_current_one_and_the_manifest_are_kept(
        self, tmp_path, monkeypatch, capsys
    ):
        home = cli_home.cli_home(_layout(tmp_path))
        for stale in ("src-" + "0" * 12, "python-stale", ".src-" + "0" * 12 + ".partial-1"):
            (home / stale / "aipotluck").mkdir(parents=True)
        _fake_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        _run(tmp_path, monkeypatch, source=source, argv=[])
        _result(capsys)

        inst = _installed(_layout(tmp_path))
        assert inst.src.is_dir()
        assert (home / cli_home.MANIFEST_NAME).exists()
        assert sorted(p.name for p in home.iterdir()) == sorted([cli_home.MANIFEST_NAME, inst.src.name])


class TestKeepSet:
    def test_a_staged_interpreter_tree_is_kept_alongside_the_source(self, tmp_path):
        home = tmp_path / "cli"
        src = home / "src-aaaa"
        inst = InstalledCli(BuildInfo(SHA_OLD, TIME_OLD), src, home / "python-cpython" / "bin" / "python3", "gui", "x")

        assert adopt.keep_set(home, inst) == {src, home / "python-cpython"}

    def test_an_interpreter_outside_the_cli_home_contributes_nothing(self, tmp_path):
        home = tmp_path / "cli"
        src = home / "src-aaaa"
        inst = InstalledCli(BuildInfo(SHA_OLD, TIME_OLD), src, tmp_path / "elsewhere" / "python3", "cli", "x")

        assert adopt.keep_set(home, inst) == {src}


# --------------------------------------------------------------- second run / versions


class TestSecondRun:
    def test_a_second_run_is_current_and_touches_neither_the_service_nor_the_shim(
        self, tmp_path, monkeypatch, capsys
    ):
        factory, mgr = _fake_manager(monkeypatch, state=ServiceState.RUNNING)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)
        _run(tmp_path, monkeypatch, source=source, argv=[])
        _result(capsys)
        factory.reset_mock()
        mgr.reset_mock()
        shim_before = _shim_text(tmp_path)

        rc, lay = _run(tmp_path, monkeypatch, source=source, argv=[])
        result = _result(capsys)

        assert rc == 0
        assert result["action"] == "current"
        assert result["repointed"] is False
        assert result["service_restarted"] is False
        factory.assert_not_called()  # no status, no stop, no install, no start
        mgr.stop.assert_not_called()
        mgr.install.assert_not_called()
        mgr.start.assert_not_called()
        assert _shim_text(tmp_path) == shim_before

    def test_an_older_tree_keeps_the_newer_install_and_leaves_the_service_alone(
        self, tmp_path, monkeypatch, capsys
    ):
        installed = _plant_manifest(_layout(tmp_path), sha=SHA_NEW, commit_time=TIME_NEW)
        factory, mgr = _fake_manager(monkeypatch, state=ServiceState.RUNNING)
        older = _fake_source(tmp_path, sha=SHA_OLD, commit_time=TIME_OLD, name="older")

        rc, lay = _run(tmp_path, monkeypatch, source=older, argv=[])
        result = _result(capsys)

        assert rc == 0
        assert result["action"] == "kept-newer"
        assert result["sha"] == SHA_OLD  # this tree's commit
        assert result["installed_sha"] == SHA_NEW  # what the CLI home still holds
        assert result["repointed"] is False
        assert result["src"] == str(installed.src)
        factory.assert_not_called()
        mgr.install.assert_not_called()
        mgr.stop.assert_not_called()
        assert _installed(lay).build == BuildInfo(SHA_NEW, TIME_NEW)
        assert not (cli_home.cli_home(lay) / f"src-{SHA_OLD[:12]}").exists()


class TestRegister:
    """cli_home.register itself: the shim is always written, the service only when one exists, and a
    stopped service is not started behind the caller's back."""

    def _inst(self, tmp_path) -> InstalledCli:
        lay = _layout(tmp_path)
        return _plant_manifest(lay, sha=SHA_NEW, commit_time=TIME_NEW)

    def test_service_args_are_the_config_and_log_dirs(self, tmp_path):
        lay = _layout(tmp_path)
        assert cli_home.service_args(lay) == [
            "--config-dir", str(lay.config_dir),
            "--log-dir", str(lay.log_dir),
        ]

    def test_a_stopped_service_is_re_registered_but_not_started(self, tmp_path, monkeypatch):
        _, mgr = _fake_manager(monkeypatch, state=ServiceState.STOPPED)
        inst = self._inst(tmp_path)

        installed, restarted, shim_path = cli_home.register(
            PROFILE, _layout(tmp_path), inst, register_service=True, start=False
        )

        assert (installed, restarted) == (True, False)
        mgr.stop.assert_not_called()
        mgr.install.assert_called_once()
        mgr.start.assert_not_called()
        assert shim_path.exists()
        assert str(inst.src / "aipotluck" / "installer" / "cli.py") in shim_path.read_text()

    def test_start_true_starts_a_stopped_service_without_calling_it_restarted(self, tmp_path, monkeypatch):
        _, mgr = _fake_manager(monkeypatch, state=ServiceState.STOPPED)
        inst = self._inst(tmp_path)

        installed, restarted, _ = cli_home.register(
            PROFILE, _layout(tmp_path), inst, register_service=True, start=True
        )

        assert (installed, restarted) == (True, False)
        mgr.start.assert_called_once_with(cli_home.SERVICE_NAME, system_scope=False)

    def test_register_service_false_never_looks_for_a_service_manager(self, tmp_path, monkeypatch):
        factory, mgr = _fake_manager(monkeypatch, state=ServiceState.RUNNING)
        inst = self._inst(tmp_path)

        installed, restarted, shim_path = cli_home.register(
            PROFILE, _layout(tmp_path), inst, register_service=False, start=True
        )

        assert (installed, restarted) == (False, False)
        factory.assert_not_called()
        mgr.install.assert_not_called()
        assert shim_path.exists()

    def test_a_failed_start_is_a_warning_and_never_called_restarted(self, tmp_path, monkeypatch, caplog):
        _, mgr = _fake_manager(
            monkeypatch, state=ServiceState.RUNNING, start_error=RuntimeError("Failed to start aipotluck")
        )
        inst = self._inst(tmp_path)

        with caplog.at_level(logging.WARNING):
            installed, restarted, shim_path = cli_home.register(
                PROFILE, _layout(tmp_path), inst, register_service=True, start=False
            )

        assert (installed, restarted) == (True, False)
        mgr.start.assert_called_once()
        assert any(r.levelname == "WARNING" and "Failed to start" in r.getMessage() for r in caplog.records)
        assert shim_path.exists()


class TestRepointingAnExistingInstall:
    def test_current_but_a_shim_pointing_elsewhere_is_repointed(self, tmp_path, monkeypatch, capsys):
        # A shim written before the CLI home existed (or by an older installer) points at some other
        # cli.py; the commit is already current, so nothing is staged, but the shim still has to move.
        _fake_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)
        _run(tmp_path, monkeypatch, source=source, argv=[])
        _result(capsys)
        shim = tmp_path / "bin" / cli_shim.CLI_SHIM_NAME
        shim.write_text('#!/usr/bin/env bash\nexec "/old/python3" "/old/tree/aipotluck/installer/cli.py" "$@"\n')

        rc, lay = _run(tmp_path, monkeypatch, source=source, argv=[])
        result = _result(capsys)

        assert rc == 0
        assert result["action"] == "current"
        assert result["repointed"] is True
        inst = _installed(lay)
        assert str(inst.src / "aipotluck" / "installer" / "cli.py") in _shim_text(tmp_path)
        assert "/old/tree" not in _shim_text(tmp_path)

    def test_no_registered_service_is_never_created_but_the_shim_is_written(
        self, tmp_path, monkeypatch, capsys
    ):
        # The GUI-hosted case: nothing registered with the OS, so adopt must not register a unit --
        # but the person's shell still needs a working command.
        _, mgr = _fake_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        rc, lay = _run(tmp_path, monkeypatch, source=source, argv=[])
        result = _result(capsys)

        assert rc == 0
        mgr.status.assert_called_once_with(cli_home.SERVICE_NAME)
        mgr.install.assert_not_called()
        mgr.stop.assert_not_called()
        mgr.start.assert_not_called()
        assert result["service_restarted"] is False
        assert (tmp_path / "bin" / cli_shim.CLI_SHIM_NAME).exists()
        assert str(_installed(lay).src) in _shim_text(tmp_path)

    def test_a_failed_registration_still_exits_zero_with_the_shim_written(
        self, tmp_path, monkeypatch, capsys, caplog
    ):
        """A service manager that refuses (Windows' Task Scheduler with no desktop session, a locked
        -down systemd) must not fail the adopt: the staged CLI and the shim are what the caller came
        for, and the result line reports that the service was not restarted."""
        _, mgr = _fake_manager(
            monkeypatch, state=ServiceState.RUNNING, install_error=RuntimeError("Access is denied.")
        )
        source = _fake_source(tmp_path, sha=SHA_NEW, commit_time=TIME_NEW)

        with caplog.at_level(logging.ERROR):
            rc, lay = _run(tmp_path, monkeypatch, source=source, argv=[])
        result = _result(capsys)

        assert rc == 0
        assert result["action"] == "adopted"
        assert result["service_restarted"] is False
        mgr.start.assert_not_called()
        assert any(r.levelname == "ERROR" and "Access is denied" in r.getMessage() for r in caplog.records)
        assert (tmp_path / "bin" / cli_shim.CLI_SHIM_NAME).exists()
        assert str(_installed(lay).src) in _shim_text(tmp_path)
