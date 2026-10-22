"""aipotluck.installer.uninstall -- what an uninstall removes, in what order, and what it never touches.

Real files in a temp layout; the only stand-ins are the OS service manager (a recorder -- the real
backends are covered by their own tests and by a real run) and llama.cpp's cache deletion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aipotluck.installer import cli, layout as layout_mod, uninstall
from aipotluck.installer.platform_detect import HostProfile
from aipotluck.installer.service.base import ServiceState, ServiceStatus


class RecordingServiceManager:
    def __init__(self, fail_stop=False, still_running=False):
        self.calls: list[str] = []
        self.fail_stop = fail_stop
        self.still_running = still_running

    def stop(self, name, *, system_scope=False):
        self.calls.append(f"stop:{name}")
        if self.fail_stop:
            raise RuntimeError("would not stop")

    def uninstall(self, name, *, system_scope=False):
        self.calls.append(f"uninstall:{name}")

    def status(self, name, *, system_scope=False):
        return ServiceStatus(ServiceState.RUNNING if self.still_running else ServiceState.STOPPED)


LINUX = HostProfile(os_name="linux", arch="x64", backend="cpu")


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A complete per-user install under tmp_path/aipotluck, the shim in tmp_path/bin, a model cache
    elsewhere, and an unrelated file beside the install root."""
    monkeypatch.setattr(uninstall.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    (tmp_path / "home").mkdir()
    root = tmp_path / "data" / "aipotluck"
    lay = layout_mod.get_layout("linux", override_root=root)
    server = root / "llama.cpp" / "b10989" / "linux-x64-cpu" / "llama-server"
    for d in (server.parent, root / "newt" / "1.18.1", lay.config_dir, lay.log_dir, lay.state_dir / "downloads"):
        d.mkdir(parents=True)
    server.write_text("#!/bin/sh\n")
    (root / "newt" / "1.18.1" / "newt").write_text("x")
    (lay.config_dir / "runtime.json").write_text(json.dumps({
        "logged_in": True, "tunnel": {"id": "t"},
        "llama_cpp": {"server_binary": str(server), "port": 1},
    }))
    (lay.state_dir / "downloads" / "llama.zip").write_text("x")
    (lay.log_dir / "aipotluck.log").write_text("x")
    gui_runtime = root / "gui-runtime"  # the GUI's own bundle lives beside; not the CLI's to delete
    gui_runtime.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "aipotluck-local-client").write_text("#!/bin/sh\n")
    monkeypatch.setattr(uninstall.cli_shim, "_bin_dir", lambda profile, system_scope: bin_dir)
    models = tmp_path / "hf-cache"
    models.mkdir()
    (models / "weights.gguf").write_text("x")
    monkeypatch.setattr(uninstall, "_STOP_TIMEOUT_SECONDS", 0.2)
    return lay, server, bin_dir, models, gui_runtime


def test_removes_service_shim_and_every_install_directory(installed):
    lay, server, bin_dir, models, gui_runtime = installed
    svc = RecordingServiceManager()
    report = uninstall.uninstall(LINUX, lay, svc)
    assert svc.calls == ["stop:aipotluck", "uninstall:aipotluck"]
    assert report.service_removed
    assert not (bin_dir / "aipotluck-local-client").exists()
    for gone in (lay.install_root / "llama.cpp", lay.install_root / "newt", lay.config_dir, lay.log_dir, lay.state_dir):
        assert not gone.exists(), gone
    # pairing went with config
    assert not (lay.config_dir / "runtime.json").exists()


def test_removes_the_path_block_from_shell_startup_files(installed, monkeypatch, tmp_path):
    lay, *_ = installed
    home = tmp_path / "home"
    original = "export EDITOR=vi\n"
    (home / ".bashrc").write_text(original)
    monkeypatch.setattr(uninstall.cli_shim, "_login_shell", lambda: "bash")
    uninstall.cli_shim.add_to_shell_profiles(tmp_path / "bin", home=home)
    assert uninstall.cli_shim.PROFILE_BEGIN in (home / ".bashrc").read_text()
    uninstall.uninstall(LINUX, lay, RecordingServiceManager())
    assert (home / ".bashrc").read_text() == original


def test_keeps_the_model_cache_unless_asked(installed, monkeypatch):
    lay, *_rest, models, _gui = installed
    monkeypatch.setattr(uninstall.cache_router, "delete", lambda *a: pytest.fail("no model deletion expected"))
    uninstall.uninstall(LINUX, lay, RecordingServiceManager())
    assert (models / "weights.gguf").exists()


def test_never_touches_what_it_did_not_install_beside_the_root(installed):
    """The GUI's own runtime sits in the same `aipotluck` directory; the GUI deletes that itself."""
    lay, *_rest, gui_runtime = installed
    uninstall.uninstall(LINUX, lay, RecordingServiceManager())
    assert gui_runtime.exists() and lay.install_root.exists()


def test_removes_every_cached_model_through_llama_cpp_before_deleting_its_binary(installed, monkeypatch):
    lay, server, *_ = installed
    seen = []
    monkeypatch.setattr(uninstall, "list_cached_models", lambda srv: ["org/a:Q4", "org/b:Q8"])
    monkeypatch.setattr(uninstall.cache_router, "delete", lambda srv, m: seen.append((m, srv.exists())))
    report = uninstall.uninstall(LINUX, lay, RecordingServiceManager(), remove_models=True)
    assert seen == [("org/a:Q4", True), ("org/b:Q8", True)]  # binary still there for each
    assert report.models_removed == ["org/a:Q4", "org/b:Q8"]
    assert not server.exists()


def test_a_model_that_cannot_be_removed_stops_before_anything_else_is_deleted(installed, monkeypatch):
    lay, server, bin_dir, *_ = installed
    monkeypatch.setattr(uninstall, "list_cached_models", lambda srv: ["org/a:Q4"])

    def refuse(srv, m):
        raise uninstall.cache_router.CacheRouterError("in use")

    monkeypatch.setattr(uninstall.cache_router, "delete", refuse)
    svc = RecordingServiceManager()
    with pytest.raises(uninstall.UninstallError, match="Run uninstall again"):
        uninstall.uninstall(LINUX, lay, svc, remove_models=True)
    assert server.exists() and (lay.config_dir / "runtime.json").exists()
    assert "uninstall:aipotluck" not in svc.calls  # retryable: the service registration is intact too


def test_a_service_that_wont_stop_is_still_removed_and_reported(installed):
    lay, *_ = installed
    svc = RecordingServiceManager(fail_stop=True)
    report = uninstall.uninstall(LINUX, lay, svc)
    assert "uninstall:aipotluck" in svc.calls
    assert any("would not stop" in w for w in report.warnings)


def test_refuses_while_the_service_is_still_running_and_removes_nothing(installed):
    lay, server, *_ = installed
    svc = RecordingServiceManager(still_running=True)
    with pytest.raises(uninstall.UninstallError, match="still running"):
        uninstall.uninstall(LINUX, lay, svc)
    assert server.exists() and "uninstall:aipotluck" not in svc.calls


def test_another_process_on_the_same_port_does_not_block_it(installed):
    """Found on a real Windows box: the desktop user's own install served on the port this test
    install would use, and a port check took that for this install still running."""
    import socket
    lay, server, *_ = installed
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        cfg = json.loads((lay.config_dir / "runtime.json").read_text())
        cfg["llama_cpp"]["port"] = busy.getsockname()[1]
        (lay.config_dir / "runtime.json").write_text(json.dumps(cfg))
        uninstall.uninstall(LINUX, lay, RecordingServiceManager())
    assert not server.exists()


def test_an_install_dir_outside_an_aipotluck_directory_is_left_alone(tmp_path, monkeypatch):
    """--install-dir is the one way a caller could aim this somewhere else: deleting `config`,
    `logs`, `state` under the user's home would be catastrophic."""
    monkeypatch.setattr(uninstall.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    home = tmp_path / "home"
    (home / "config").mkdir(parents=True)
    (home / "config" / "precious").write_text("x")
    monkeypatch.setattr(uninstall.cli_shim, "_bin_dir", lambda profile, system_scope: tmp_path / "bin")
    report = uninstall.uninstall(LINUX, layout_mod.get_layout("linux", override_root=home), RecordingServiceManager())
    assert (home / "config" / "precious").exists()
    assert any("left" in w for w in report.warnings)


@pytest.mark.parametrize("path, ok", [
    ("/", False), ("{home}", False), ("{home}/aipotluck", True), ("{home}/aipotluck/config", True),
    ("{home}/.config", False), ("{home}/x/aipotluck/logs", True),
])
def test_safe_to_delete(tmp_path, monkeypatch, path, ok):
    monkeypatch.setattr(uninstall.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert uninstall._safe_to_delete(Path(path.format(home=tmp_path / "home"))) is ok


def test_cli_refuses_without_yes_when_there_is_no_terminal(installed, monkeypatch):
    lay, server, *_ = installed
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    args = cli.build_arg_parser().parse_args(["uninstall", "--install-dir", str(lay.install_root)])
    assert cli.run_uninstall(args) == 1
    assert server.exists()


def test_cli_with_yes_uninstalls(installed, monkeypatch):
    lay, server, *_ = installed
    svc = RecordingServiceManager()
    monkeypatch.setattr(cli, "get_service_manager", lambda os_name: svc)
    monkeypatch.setattr(cli, "detect_host_profile", lambda *a: LINUX)
    args = cli.build_arg_parser().parse_args(["uninstall", "--yes", "--install-dir", str(lay.install_root)])
    assert cli.run_uninstall(args) == 0
    assert not server.exists() and svc.calls[-1] == "uninstall:aipotluck"


def test_models_only_removes_each_model_the_way_remove_does_and_leaves_the_install(installed, monkeypatch):
    lay, server, bin_dir, *_ = installed
    from aipotluck.service import model_ops
    monkeypatch.setattr(uninstall, "list_cached_models", lambda srv: ["org/a:Q4", "org/b:Q8"])
    calls = []
    monkeypatch.setattr(model_ops, "remove", lambda cfg, config_dir, m: calls.append((m, config_dir)))
    assert uninstall.remove_all_models(lay) == ["org/a:Q4", "org/b:Q8"]
    assert calls == [("org/a:Q4", lay.config_dir), ("org/b:Q8", lay.config_dir)]
    assert server.exists() and (lay.config_dir / "runtime.json").exists() and (bin_dir / "aipotluck-local-client").exists()


def test_models_only_with_nothing_installed_says_so(tmp_path):
    lay = layout_mod.get_layout("linux", override_root=tmp_path / "aipotluck")
    with pytest.raises(uninstall.UninstallError, match="nothing is installed"):
        uninstall.remove_all_models(lay)


def test_cli_models_only_never_touches_the_service(installed, monkeypatch):
    lay, server, *_ = installed
    monkeypatch.setattr(cli, "get_service_manager", lambda os_name: pytest.fail("service must be left alone"))
    monkeypatch.setattr(cli, "detect_host_profile", lambda *a: LINUX)
    monkeypatch.setattr(uninstall, "list_cached_models", lambda srv: [])
    args = cli.build_arg_parser().parse_args(["uninstall", "--models-only", "--yes", "--install-dir", str(lay.install_root)])
    assert cli.run_uninstall(args) == 0
    assert server.exists()
