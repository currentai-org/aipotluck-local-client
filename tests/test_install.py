"""aipotluck.installer.install -- the one-shot installer.

fetch.* (no real network download), get_service_manager (no real systemd/launchd), ensure_python,
and install_cli_shim are all mocked -- these tests pin the CONTRACT (what runtime.json ends up
containing, what gets called with what) rather than re-proving llama.cpp download/extraction,
which fetch.py's own responsibility.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aipotluck.installer import cli_home, cli_shim, install
from aipotluck.installer.cli_home import BuildInfo, InstalledCli
from aipotluck.installer.layout import Layout
from aipotluck.installer.platform_detect import HostProfile
from aipotluck.installer.service.base import ServiceState, ServiceStatus


@pytest.fixture(autouse=True)
def _stable_profile(monkeypatch):
    monkeypatch.setattr(
        install, "detect_host_profile", lambda backend: HostProfile(os_name="linux", arch="x64", backend="cpu")
    )


@pytest.fixture(autouse=True)
def _fake_model_prefetch(monkeypatch):
    """The default-model pre-fetch/pre-size step (router-mode auto-sizing) is a real
    subprocess/network boundary, same category as fetch.*/get_service_manager above -- mocked by
    default so every other test in this file isn't incidentally exercising it against a fake,
    non-executable placeholder binary. TestRunInstallDefaultModelSizing below un-mocks it
    selectively to pin the actual contract."""
    pull_calls = []
    ensure_preset_calls = []
    monkeypatch.setattr(install.model_pull, "pull_model", lambda *a, **kw: pull_calls.append((a, kw)))
    monkeypatch.setattr(
        install.model_sizing, "ensure_preset", lambda *a, **kw: ensure_preset_calls.append((a, kw))
    )
    return pull_calls, ensure_preset_calls


@pytest.fixture(autouse=True)
def _never_a_real_os_service(monkeypatch):
    """No test in this file may register (or start, or stop) a real OS service. cli_home's manager
    seam is what the per-user branch registers through, so it is a MagicMock by default: even a
    regression that routes a test down a path it wasn't meant to take writes nothing to the real
    systemd/launchd/Task Scheduler. Tests that assert on the registration replace this with their
    own mock."""
    mgr = MagicMock()
    mgr.status.return_value = ServiceStatus(ServiceState.NOT_INSTALLED, detail="fake")
    factory = MagicMock(return_value=mgr)
    monkeypatch.setattr(cli_home, "get_service_manager", factory)
    return factory, mgr


@pytest.fixture
def fake_fetch(monkeypatch, tmp_path):
    """Stubs the whole download+extract pipeline. find_binary's return value is a real file (some
    downstream code, e.g. the service-registration step, could plausibly check existence even
    though run_install itself doesn't) so nothing here is checksum/network-shaped, just present."""
    server_binary = tmp_path / "extracted" / "llama-server"
    server_binary.parent.mkdir(parents=True, exist_ok=True)
    server_binary.write_text("#!/bin/sh\n")

    monkeypatch.setattr(install.fetch, "load_version_manifest", lambda path: {"tag": "b10989"})
    monkeypatch.setattr(
        install.build_strategy,
        "resolve_install_strategy",
        lambda profile, manifest: install.build_strategy.InstallStrategy(
            use_source_build=False, reason="test", asset_key="linux-x64-cpu", asset_entry={"file": "llama-x.tar.gz"}
        ),
    )
    monkeypatch.setattr(install.fetch, "download_asset", lambda manifest, entry, dest_dir: tmp_path / "archive.tar.gz")
    monkeypatch.setattr(
        install.fetch, "extract_archive", lambda archive, root, tag, asset_key: server_binary.parent
    )
    monkeypatch.setattr(install.fetch, "find_binary", lambda extract_dir, stem: server_binary)
    return server_binary


@pytest.fixture
def fake_source_build(tmp_path, monkeypatch):
    """Stubs the strategy resolver to force the source-build path, plus the build itself -- these
    tests pin run_install's CONTRACT around that path (what it refuses without asking, what it
    passes through to source_build.ensure_llama_server_built), not source_build.py's own behavior
    (that's test_source_build.py's job)."""
    server_binary = tmp_path / "built" / "bin" / "llama-server"
    server_binary.parent.mkdir(parents=True, exist_ok=True)
    server_binary.write_text("#!/bin/sh\n")

    monkeypatch.setattr(install.fetch, "load_version_manifest", lambda path: {"tag": "b10989"})
    monkeypatch.setattr(
        install.build_strategy,
        "resolve_install_strategy",
        lambda profile, manifest: install.build_strategy.InstallStrategy(
            use_source_build=True, reason="no published asset", cuda_arch="87"
        ),
    )
    monkeypatch.setattr(install.source_build, "check_build_prerequisites", lambda backend: [])
    monkeypatch.setattr(install.source_build, "check_free_disk_gb", lambda path: 20.0)
    ensure_built = MagicMock(return_value=server_binary)
    monkeypatch.setattr(install.source_build, "ensure_llama_server_built", ensure_built)
    return server_binary, ensure_built


def make_args(install_dir: Path, extra_argv: list[str] | None = None):
    argv = ["--install-dir", str(install_dir), "--model-path", str(install_dir / "model.gguf")]
    argv += extra_argv or []
    return install.build_arg_parser().parse_args(argv)


class TestArgParsing:
    def test_model_hf_and_model_path_are_mutually_exclusive(self):
        parser = install.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model-hf", "org/repo", "--model-path", "/x.gguf"])

    def test_default_model_hf_is_the_placeholder(self):
        parser = install.build_arg_parser()
        args = parser.parse_args([])
        assert args.model_hf == install.DEFAULT_MODEL_HF
        assert args.model_path is None

    def test_backend_choices_are_restricted(self):
        parser = install.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--backend", "quantum"])


class TestReinstallKeepsPairing:
    """Re-running the installer is the normal way to pick up a newer llama.cpp build, so it must
    not cost the device its pairing.

    The failure it guards is quiet in the worst way: the running service keeps serving from the
    config it already holds in memory, so nothing looks wrong until the next restart -- possibly
    days later, and looking entirely unrelated to the install. Seen on a Jetson that went dark at
    its next restart after being re-installed.
    """

    def _runtime(self, install_dir: Path) -> dict:
        return json.loads((install_dir / "config" / "runtime.json").read_text())

    def _pair(self, install_dir: Path) -> None:
        path = install_dir / "config" / "runtime.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "llama_cpp": {"server_binary": "/old/llama-server"},
            "logged_in": True,
            "tunnel": {"provider": "pangolin", "id": "abc", "secret": "shh", "endpoint": "https://x"},
        }), encoding="utf-8")

    def test_an_existing_pairing_survives_a_reinstall(self, tmp_path, fake_fetch):
        install_dir = tmp_path / "install"
        self._pair(install_dir)

        assert install.run_install(make_args(install_dir, ["--no-service"])) == 0

        saved = self._runtime(install_dir)
        assert saved["logged_in"] is True
        assert saved["tunnel"]["id"] == "abc" and saved["tunnel"]["secret"] == "shh"

    def test_the_llama_section_is_still_rebuilt(self, tmp_path, fake_fetch):
        """Only the pairing is carried over. The llama.cpp paths describe what was just installed,
        so keeping the old ones would point the service at a binary that may no longer be there."""
        install_dir = tmp_path / "install"
        self._pair(install_dir)

        install.run_install(make_args(install_dir, ["--no-service"]))

        assert self._runtime(install_dir)["llama_cpp"]["server_binary"] == str(fake_fetch)

    def test_a_logged_out_device_stays_logged_out(self, tmp_path, fake_fetch):
        install_dir = tmp_path / "install"
        (install_dir / "config").mkdir(parents=True, exist_ok=True)
        (install_dir / "config" / "runtime.json").write_text(
            json.dumps({"logged_in": False}), encoding="utf-8"
        )

        install.run_install(make_args(install_dir, ["--no-service"]))

        saved = self._runtime(install_dir)
        assert saved["logged_in"] is False and "tunnel" not in saved

    def test_a_claimed_login_with_no_tunnel_is_not_honoured(self, tmp_path, fake_fetch):
        """`logged_in` without credentials cannot serve anything, so carrying it over would leave
        the service trying to start a tunnel it has no details for."""
        install_dir = tmp_path / "install"
        (install_dir / "config").mkdir(parents=True, exist_ok=True)
        (install_dir / "config" / "runtime.json").write_text(
            json.dumps({"logged_in": True}), encoding="utf-8"
        )

        install.run_install(make_args(install_dir, ["--no-service"]))

        assert self._runtime(install_dir)["logged_in"] is False

    def test_a_corrupt_runtime_config_does_not_stop_the_install(self, tmp_path, fake_fetch):
        install_dir = tmp_path / "install"
        (install_dir / "config").mkdir(parents=True, exist_ok=True)
        (install_dir / "config" / "runtime.json").write_text("{not json", encoding="utf-8")

        assert install.run_install(make_args(install_dir, ["--no-service"])) == 0
        assert self._runtime(install_dir)["logged_in"] is False


class TestRunInstallNoService:
    def test_writes_logged_out_runtime_config(self, tmp_path, fake_fetch, capsys):
        install_dir = tmp_path / "install"
        args = make_args(install_dir, ["--no-service"])

        rc = install.run_install(args)

        assert rc == 0
        runtime_path = install_dir / "config" / "runtime.json"
        saved = json.loads(runtime_path.read_text())
        assert saved["logged_in"] is False
        assert "tunnel" not in saved
        assert saved["llama_cpp"]["server_binary"] == str(fake_fetch)
        # Router mode: no single "active model" in runtime.json anymore -- ctx_size/
        # model_hf/model_path all moved to being per-model, in the --models-preset INI file.
        assert "model_hf" not in saved["llama_cpp"]
        assert "model_path" not in saved["llama_cpp"]
        assert "ctx_size" not in saved["llama_cpp"]
        assert saved["llama_cpp"]["models_dir"] == str(install_dir / "model.gguf").rsplit("/", 1)[0]
        assert saved["llama_cpp"]["presets_path"] == str(install_dir / "config" / install.PRESETS_FILENAME)
        assert saved["llama_cpp"]["models_max"] == install.DEFAULT_MODELS_MAX

    def test_skips_service_and_shim(self, tmp_path, fake_fetch, monkeypatch):
        get_service_manager = MagicMock()
        install_cli_shim = MagicMock()
        monkeypatch.setattr(install, "get_service_manager", get_service_manager)
        monkeypatch.setattr(install, "install_cli_shim", install_cli_shim)
        args = make_args(tmp_path / "install", ["--no-service"])

        install.run_install(args)

        get_service_manager.assert_not_called()
        install_cli_shim.assert_not_called()

    def test_summary_mentions_module_fallback_not_the_shim(self, tmp_path, fake_fetch, capsys):
        args = make_args(tmp_path / "install", ["--no-service"])
        install.run_install(args)
        out = capsys.readouterr().out
        assert "not installed (--no-service)" in out
        assert "python3 -m aipotluck.installer.cli login" in out
        assert "CLI:" not in out


class TestRunInstallCustomBuildCache:
    def _cache_hit(self, monkeypatch, tmp_path, *, key="linux-x64-cuda-sm87"):
        server_binary = tmp_path / "custom_extracted" / "llama-server"
        server_binary.parent.mkdir(parents=True, exist_ok=True)
        server_binary.write_text("#!/bin/sh\n")

        monkeypatch.setattr(
            install, "detect_host_profile",
            lambda backend: HostProfile(os_name="linux", arch="x64", backend="cuda"),
        )
        monkeypatch.setattr(install.fetch, "load_version_manifest", lambda path: {"tag": "b10989"})
        monkeypatch.setattr(
            install.build_strategy, "resolve_install_strategy",
            lambda profile, manifest: install.build_strategy.InstallStrategy(
                use_source_build=True, reason="no published asset", cuda_arch="87"
            ),
        )
        monkeypatch.setattr(
            install.build_cache, "load_custom_manifest",
            lambda path: {"tag": "custom-builds", "assets": {key: {"file": "f.tar.gz", "sha256": "abc"}}},
        )
        monkeypatch.setattr(install.fetch, "download_asset", lambda manifest, entry, dest_dir: tmp_path / "f.tar.gz")
        monkeypatch.setattr(
            install.fetch, "extract_archive", lambda archive, root, tag, asset_key: server_binary.parent
        )
        monkeypatch.setattr(install.fetch, "find_binary", lambda extract_dir, stem: server_binary)
        return server_binary

    def test_cache_hit_skips_compilation_entirely(self, tmp_path, monkeypatch):
        server_binary = self._cache_hit(monkeypatch, tmp_path)
        ensure_built = MagicMock()
        monkeypatch.setattr(install.source_build, "ensure_llama_server_built", ensure_built)
        check_prereqs = MagicMock()
        monkeypatch.setattr(install.source_build, "check_build_prerequisites", check_prereqs)
        args = make_args(tmp_path / "install", ["--no-service"])

        rc = install.run_install(args)

        assert rc == 0
        ensure_built.assert_not_called()
        check_prereqs.assert_not_called()  # no preflight either -- a cache hit isn't a build at all
        saved = json.loads((tmp_path / "install" / "config" / "runtime.json").read_text())
        assert saved["llama_cpp"]["server_binary"] == str(server_binary)
        assert saved["llama_cpp"]["asset_key"] == "custom-build:linux-x64-cuda-sm87"
        assert saved["llama_cpp"]["built_from_source"] is False  # downloaded, not compiled on this host

    def test_cache_hit_ignores_no_source_build_flag(self, tmp_path, monkeypatch):
        # --no-source-build means "don't compile"; a cache download isn't a compile, so it must
        # still succeed even with this flag set.
        self._cache_hit(monkeypatch, tmp_path)
        args = make_args(tmp_path / "install", ["--no-service", "--no-source-build"])

        rc = install.run_install(args)

        assert rc == 0

    def test_no_cache_entry_falls_through_to_a_real_build(self, tmp_path, fake_source_build, monkeypatch):
        _, ensure_built = fake_source_build
        monkeypatch.setattr(install.build_cache, "load_custom_manifest", lambda path: {"assets": {}})
        args = make_args(tmp_path / "install", ["--no-service"])

        rc = install.run_install(args)

        assert rc == 0
        ensure_built.assert_called_once()


class TestRunInstallSourceBuild:
    def test_builds_from_source_and_records_it_in_runtime_config(self, tmp_path, fake_source_build):
        server_binary, ensure_built = fake_source_build
        install_dir = tmp_path / "install"
        args = make_args(install_dir, ["--no-service"])

        rc = install.run_install(args)

        assert rc == 0
        ensure_built.assert_called_once()
        _, kwargs = ensure_built.call_args
        assert kwargs["cuda_arch"] == "87"
        saved = json.loads((install_dir / "config" / "runtime.json").read_text())
        assert saved["llama_cpp"]["server_binary"] == str(server_binary)
        assert saved["llama_cpp"]["built_from_source"] is True

    def test_no_source_build_flag_refuses_instead_of_building(self, tmp_path, fake_source_build):
        _, ensure_built = fake_source_build
        args = make_args(tmp_path / "install", ["--no-service", "--no-source-build"])

        rc = install.run_install(args)

        assert rc == 1
        ensure_built.assert_not_called()

    def test_missing_prerequisites_refuses_before_building(self, tmp_path, fake_source_build, monkeypatch):
        _, ensure_built = fake_source_build
        monkeypatch.setattr(
            install.source_build, "check_build_prerequisites",
            lambda backend: ["cmake not found (sudo apt-get install -y cmake)"],
        )
        args = make_args(tmp_path / "install", ["--no-service"])

        rc = install.run_install(args)

        assert rc == 1
        ensure_built.assert_not_called()

    def test_insufficient_disk_refuses_before_building(self, tmp_path, fake_source_build, monkeypatch):
        _, ensure_built = fake_source_build
        monkeypatch.setattr(install.source_build, "check_free_disk_gb", lambda path: 1.0)
        args = make_args(tmp_path / "install", ["--no-service"])

        rc = install.run_install(args)

        assert rc == 1
        ensure_built.assert_not_called()

    def test_cuda_backend_with_undetectable_arch_refuses_rather_than_guessing(
        self, tmp_path, fake_source_build, monkeypatch
    ):
        _, ensure_built = fake_source_build
        monkeypatch.setattr(
            install, "detect_host_profile",
            lambda backend: HostProfile(os_name="linux", arch="arm64", backend="cuda"),
        )
        monkeypatch.setattr(
            install.build_strategy, "resolve_install_strategy",
            lambda profile, manifest: install.build_strategy.InstallStrategy(
                use_source_build=True, reason="no published asset", cuda_arch=None
            ),
        )
        args = make_args(tmp_path / "install", ["--no-service"])

        rc = install.run_install(args)

        assert rc == 1
        ensure_built.assert_not_called()

    def test_jobs_and_timeout_flags_pass_through(self, tmp_path, fake_source_build):
        _, ensure_built = fake_source_build
        args = make_args(tmp_path / "install", ["--no-service", "--jobs", "3", "--build-timeout", "120"])

        install.run_install(args)

        _, kwargs = ensure_built.call_args
        assert kwargs["jobs"] == 3
        assert kwargs["timeout"] == 120.0


class TestConfirmAptInstall:
    def test_no_tty_never_prompts_and_declines(self, monkeypatch):
        monkeypatch.setattr(install.sys.stdin, "isatty", lambda: False)
        assert install._confirm_apt_install(["cmake"]) is False

    def test_tty_yes_confirms(self, monkeypatch):
        monkeypatch.setattr(install.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt: "y")
        assert install._confirm_apt_install(["cmake"]) is True

    def test_tty_blank_declines(self, monkeypatch):
        monkeypatch.setattr(install.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt: "")
        assert install._confirm_apt_install(["cmake"]) is False

    def test_eof_declines_rather_than_crashing(self, monkeypatch):
        monkeypatch.setattr(install.sys.stdin, "isatty", lambda: True)

        def _raise(prompt):
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise)
        assert install._confirm_apt_install(["cmake"]) is False


class TestRunInstallAptInstallConsent:
    def _missing_prereqs(self, monkeypatch, *, packages, resolved_by_install=True):
        """Simulates check_build_prerequisites reporting `packages` missing on its first call --
        as if install_apt_packages (mocked separately per test) had actually fixed them, a second
        call comes back clean when `resolved_by_install` is True, or still missing when it's not
        (e.g. the "apt install itself failed" test, where nothing was ever really fixed)."""
        calls = {"n": 0}

        def _check(backend):
            calls["n"] += 1
            if calls["n"] == 1 or not resolved_by_install:
                return [f"{p} not found (sudo apt-get install -y {p})" for p in packages]
            return []

        monkeypatch.setattr(install.source_build, "check_build_prerequisites", _check)
        monkeypatch.setattr(install.source_build, "missing_apt_packages", lambda: list(packages))
        monkeypatch.setattr(install.source_build, "has_apt", lambda: True)

    def test_allow_apt_install_flag_installs_without_prompting(self, tmp_path, fake_source_build, monkeypatch):
        _, ensure_built = fake_source_build
        self._missing_prereqs(monkeypatch, packages=["cmake"])
        install_apt = MagicMock()
        monkeypatch.setattr(install.source_build, "install_apt_packages", install_apt)
        confirm = MagicMock()
        monkeypatch.setattr(install, "_confirm_apt_install", confirm)
        args = make_args(tmp_path / "install", ["--no-service", "--allow-apt-install"])

        rc = install.run_install(args)

        assert rc == 0
        install_apt.assert_called_once_with(["cmake"])
        confirm.assert_not_called()  # explicit consent up front skips the interactive prompt entirely
        ensure_built.assert_called_once()

    def test_no_apt_install_flag_never_offers_even_interactively(self, tmp_path, fake_source_build, monkeypatch):
        _, ensure_built = fake_source_build
        self._missing_prereqs(monkeypatch, packages=["cmake"])
        install_apt = MagicMock()
        monkeypatch.setattr(install.source_build, "install_apt_packages", install_apt)
        confirm = MagicMock(return_value=True)
        monkeypatch.setattr(install, "_confirm_apt_install", confirm)
        args = make_args(tmp_path / "install", ["--no-service", "--no-apt-install"])

        rc = install.run_install(args)

        assert rc == 1
        install_apt.assert_not_called()
        confirm.assert_not_called()
        ensure_built.assert_not_called()

    def test_default_prompts_interactively_and_proceeds_on_yes(self, tmp_path, fake_source_build, monkeypatch):
        _, ensure_built = fake_source_build
        self._missing_prereqs(monkeypatch, packages=["cmake", "libssl-dev"])
        install_apt = MagicMock()
        monkeypatch.setattr(install.source_build, "install_apt_packages", install_apt)
        monkeypatch.setattr(install, "_confirm_apt_install", lambda packages: True)
        args = make_args(tmp_path / "install", ["--no-service"])

        rc = install.run_install(args)

        assert rc == 0
        install_apt.assert_called_once_with(["cmake", "libssl-dev"])
        ensure_built.assert_called_once()

    def test_default_declines_on_no_and_refuses_to_build(self, tmp_path, fake_source_build, monkeypatch):
        _, ensure_built = fake_source_build
        self._missing_prereqs(monkeypatch, packages=["cmake"])
        install_apt = MagicMock()
        monkeypatch.setattr(install.source_build, "install_apt_packages", install_apt)
        monkeypatch.setattr(install, "_confirm_apt_install", lambda packages: False)
        args = make_args(tmp_path / "install", ["--no-service"])

        rc = install.run_install(args)

        assert rc == 1
        install_apt.assert_not_called()
        ensure_built.assert_not_called()

    def test_apt_install_failure_is_reported_and_never_builds_anyway(self, tmp_path, fake_source_build, monkeypatch):
        _, ensure_built = fake_source_build
        self._missing_prereqs(monkeypatch, packages=["cmake"], resolved_by_install=False)
        monkeypatch.setattr(
            install.source_build, "install_apt_packages",
            MagicMock(side_effect=install.source_build.AptInstallError("wrong password")),
        )
        args = make_args(tmp_path / "install", ["--no-service", "--allow-apt-install"])

        rc = install.run_install(args)

        assert rc == 1
        ensure_built.assert_not_called()

    def test_nvcc_gap_is_never_apt_installable_even_with_consent(self, tmp_path, fake_source_build, monkeypatch):
        # missing_apt_packages() never includes nvcc (see its docstring) -- so even full consent
        # can't hide a real nvcc gap; the CUDA toolkit is never something this auto-installs.
        monkeypatch.setattr(
            install.source_build, "check_build_prerequisites",
            lambda backend: ["nvcc not found -- ..."],
        )
        monkeypatch.setattr(install.source_build, "missing_apt_packages", lambda: [])
        install_apt = MagicMock()
        monkeypatch.setattr(install.source_build, "install_apt_packages", install_apt)
        _, ensure_built = fake_source_build
        args = make_args(tmp_path / "install", ["--no-service", "--allow-apt-install"])

        rc = install.run_install(args)

        assert rc == 1
        install_apt.assert_not_called()
        ensure_built.assert_not_called()

    def test_allow_and_no_apt_install_are_mutually_exclusive(self):
        parser = install.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--allow-apt-install", "--no-apt-install"])


class TestRunInstallWithService:
    def _fake_service_manager(self, monkeypatch):
        mgr = MagicMock()
        mgr.status.return_value = ServiceStatus(ServiceState.RUNNING, detail="active")
        monkeypatch.setattr(install, "get_service_manager", lambda os_name: mgr)
        return mgr

    def test_registers_service_pointed_at_the_aipotluck_package_script(self, tmp_path, fake_fetch, monkeypatch):
        mgr = self._fake_service_manager(monkeypatch)
        monkeypatch.setattr(install, "ensure_python", lambda: Path("/usr/bin/python3"))
        monkeypatch.setattr(install, "install_cli_shim", lambda profile, python_exe, system_scope: Path("/fake/bin/aipotluck-local-client"))
        args = make_args(tmp_path / "install")

        rc = install.run_install(args)

        assert rc == 0
        _, kwargs = mgr.install.call_args
        expected_script = install.REPO_ROOT / "aipotluck" / "service" / "aipotluck_service.py"
        assert kwargs["args"][0] == str(expected_script)
        assert "--config-dir" in kwargs["args"]

    def test_installs_the_cli_shim_with_the_resolved_interpreter(self, tmp_path, fake_fetch, monkeypatch):
        self._fake_service_manager(monkeypatch)
        monkeypatch.setattr(install, "ensure_python", lambda: Path("/usr/bin/python3.12"))
        install_cli_shim = MagicMock(return_value=Path("/fake/bin/aipotluck-local-client"))
        monkeypatch.setattr(install, "install_cli_shim", install_cli_shim)
        args = make_args(tmp_path / "install")

        install.run_install(args)

        install_cli_shim.assert_called_once()
        call_profile, call_python, call_system = install_cli_shim.call_args[0]
        assert call_python == Path("/usr/bin/python3.12")
        assert call_system is False

    def test_no_start_skips_starting_the_service(self, tmp_path, fake_fetch, monkeypatch):
        mgr = self._fake_service_manager(monkeypatch)
        monkeypatch.setattr(install, "ensure_python", lambda: Path("/usr/bin/python3"))
        monkeypatch.setattr(install, "install_cli_shim", lambda *a: Path("/fake/bin/aipotluck-local-client"))
        args = make_args(tmp_path / "install", ["--no-start"])

        install.run_install(args)

        mgr.start.assert_not_called()

    def test_a_failed_service_registration_is_reported_and_never_started_but_the_install_still_completes(
        self, tmp_path, fake_fetch, monkeypatch, caplog
    ):
        """A service manager that can't register (Windows' Task Scheduler refusing an SSH logon)
        must not be reported as installed, nor 'started'; llama.cpp, runtime.json and the CLI shim
        are still usable, and the hosted fallback (the GUI) can run the service itself."""
        mgr = self._fake_service_manager(monkeypatch)
        mgr.install.side_effect = RuntimeError("Couldn't register the AIPotluck_aipotluck Scheduled Task: ERROR: Access is denied.")
        mgr.status.return_value = ServiceStatus(ServiceState.NOT_INSTALLED, detail="not found")
        monkeypatch.setattr(install, "ensure_python", lambda: Path("/usr/bin/python3"))
        shim = MagicMock(return_value=Path("/fake/bin/aipotluck-local-client"))
        monkeypatch.setattr(install, "install_cli_shim", shim)
        args = make_args(tmp_path / "install")

        with caplog.at_level("INFO"):
            rc = install.run_install(args)

        assert rc == 0
        mgr.start.assert_not_called()
        shim.assert_called_once()
        assert any(r.levelname == "ERROR" and "Access is denied" in r.getMessage() for r in caplog.records)
        assert not any("Installed aipotluck service" in r.getMessage() for r in caplog.records)

    def test_summary_shows_shim_path_and_login_command(self, tmp_path, fake_fetch, monkeypatch, capsys):
        self._fake_service_manager(monkeypatch)
        monkeypatch.setattr(install, "ensure_python", lambda: Path("/usr/bin/python3"))
        shim_path = Path("/fake/bin/aipotluck-local-client")
        monkeypatch.setattr(install, "install_cli_shim", lambda *a: shim_path)
        args = make_args(tmp_path / "install")

        install.run_install(args)

        out = capsys.readouterr().out
        assert f"CLI:            {shim_path}" in out
        assert f"{install.CLI_SHIM_NAME} login" in out
        assert "python3 -m aipotluck.installer.cli login" not in out


class TestRunInstallDefaultModelSizing:
    """Router mode: a fresh install pre-fetches (HF only) and pre-sizes the
    default model, so it has a correctly-sized preset waiting rather than falling back to
    llama-server's stock defaults until the person runs `pull`. model_pull.pull_model/
    model_sizing.ensure_preset are mocked here too (real subprocess coverage lives in
    test_model_pull.py/test_model_sizing.py) -- what belongs here is run_install's own wiring:
    what gets called, with what, and that a failure here never fails the install.
    """

    def test_model_hf_default_is_pulled_then_sized(self, tmp_path, fake_fetch, _fake_model_prefetch):
        pull_calls, ensure_preset_calls = _fake_model_prefetch
        args = install.build_arg_parser().parse_args(
            ["--install-dir", str(tmp_path / "install"), "--no-service", "--model-hf", "org/repo:Q4_K_M"]
        )

        rc = install.run_install(args)

        assert rc == 0
        assert len(pull_calls) == 1
        (server_bin, hf_target), _ = pull_calls[0]
        assert server_bin == fake_fetch
        assert hf_target == "org/repo:Q4_K_M"
        assert len(ensure_preset_calls) == 1
        _, kwargs = ensure_preset_calls[0]
        assert kwargs["model_hf"] == "org/repo:Q4_K_M"
        assert kwargs["model_path"] is None
        assert kwargs["force"] is True

    def test_model_path_is_sized_but_never_pulled(self, tmp_path, fake_fetch, _fake_model_prefetch):
        # A local GGUF is already on disk -- there is nothing to download.
        pull_calls, ensure_preset_calls = _fake_model_prefetch
        args = make_args(tmp_path / "install", ["--no-service"])  # make_args defaults to --model-path

        rc = install.run_install(args)

        assert rc == 0
        assert pull_calls == []
        assert len(ensure_preset_calls) == 1
        _, kwargs = ensure_preset_calls[0]
        assert kwargs["model_hf"] is None
        assert kwargs["model_path"] == tmp_path / "install" / "model.gguf"

    def test_no_model_pull_flag_skips_pre_fetch_and_sizing_entirely(self, tmp_path, fake_fetch, _fake_model_prefetch):
        pull_calls, ensure_preset_calls = _fake_model_prefetch
        args = make_args(tmp_path / "install", ["--no-service", "--no-model-pull"])

        rc = install.run_install(args)

        assert rc == 0
        assert pull_calls == []
        assert ensure_preset_calls == []

    def test_pull_failure_does_not_fail_the_install(self, tmp_path, fake_fetch, monkeypatch, caplog):
        def raise_pull_error(*a, **kw):
            raise install.model_pull.ModelPullError("bad repo/quant")

        monkeypatch.setattr(install.model_pull, "pull_model", raise_pull_error)
        args = install.build_arg_parser().parse_args(
            ["--install-dir", str(tmp_path / "install"), "--no-service", "--model-hf", "org/bad:Q4"]
        )

        with caplog.at_level(logging.WARNING, logger="aipotluck.installer"):
            rc = install.run_install(args)

        assert rc == 0  # the install itself still succeeds
        assert any("Could not pre-fetch/size" in rec.message for rec in caplog.records)

    def test_sizing_failure_does_not_fail_the_install(self, tmp_path, fake_fetch, monkeypatch, caplog):
        def raise_sizing_error(*a, **kw):
            raise install.model_sizing.ModelSizingError("probe timed out")

        monkeypatch.setattr(install.model_sizing, "ensure_preset", raise_sizing_error)
        args = make_args(tmp_path / "install", ["--no-service"])

        with caplog.at_level(logging.WARNING, logger="aipotluck.installer"):
            rc = install.run_install(args)

        assert rc == 0
        assert any("Could not pre-fetch/size" in rec.message for rec in caplog.records)


class TestServiceLoggingUnderPythonw:
    def test_logging_setup_survives_having_no_stdout(self, tmp_path, monkeypatch):
        """pythonw.exe (how the Windows logon task starts the service, so no console window opens)
        gives the process no stdout: sys.stdout is None. Logging setup must still work, to the file."""
        import logging
        import sys as _sys

        from aipotluck.service import aipotluck_service

        monkeypatch.setattr(_sys, "stdout", None)
        # Under pythonw stderr is None too, so a StreamHandler(None) -- which quietly falls back to
        # sys.stderr -- breaks the first time it writes. Reproduce both.
        monkeypatch.setattr(_sys, "stderr", None)
        root = logging.getLogger()
        saved = root.handlers[:]
        root.handlers = []
        try:
            aipotluck_service.setup_logging(tmp_path)
            assert all(getattr(h, "stream", True) is not None for h in root.handlers), \
                "a console handler with no console was installed"
            logging.getLogger("aipotluck.test").info("hello from pythonw")
            for h in root.handlers:
                h.flush()
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = saved
        assert "hello from pythonw" in (tmp_path / "aipotluck.log").read_text()


class TestRunInstallCompanions:
    def test_the_installer_fetches_the_assets_companions_into_the_llama_dir(self, tmp_path, fake_fetch, monkeypatch):
        """Windows CUDA's cudart zip has to be unpacked beside llama-server by the install itself,
        not just be listed in the manifest -- otherwise the CUDA build runs on the CPU."""
        calls = []
        monkeypatch.setattr(
            install.fetch, "fetch_companions",
            lambda manifest, entry, downloads, llama_dir: calls.append((entry["file"], llama_dir)),
        )
        assert install.run_install(make_args(tmp_path / "inst", ["--no-service"])) == 0
        assert calls == [("llama-x.tar.gz", fake_fetch.parent)]


# --------------------------------------------------------- the per-user CLI home (CUR-4337)

SHA_OLD = "a" * 40
SHA_NEW = "b" * 40
TIME_OLD = 1_000_000
TIME_NEW = 1_100_000


def _layout(tmp_path: Path) -> Layout:
    """A per-user Layout rooted under tmp_path -- what get_layout would return for this HOME."""
    root = tmp_path / "install"
    return Layout(install_root=root, config_dir=root / "config", log_dir=root / "logs", state_dir=root / "state")


def _fake_source(tmp_path: Path, *, sha: str = SHA_NEW, commit_time: int = TIME_NEW, name: str = "source") -> Path:
    """A throwaway tree shaped like this install's own: the package (with the two entry points a
    staged copy has to carry) plus a stamped build-info file, so its version is the one the test
    chose rather than whatever this checkout's git HEAD happens to be."""
    root = tmp_path / name
    package = root / "aipotluck"
    (package / "installer").mkdir(parents=True)
    (package / "service").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "installer" / "cli.py").write_text("# cli\n", encoding="utf-8")
    (package / "installer" / "install.py").write_text("# install\n", encoding="utf-8")
    (package / "service" / "aipotluck_service.py").write_text("# service\n", encoding="utf-8")
    (package / "_build_info.txt").write_text(f"{sha} {commit_time}\n", encoding="utf-8")
    return root


def _parse(argv: list[str] | None = None):
    """Parse as the CLI would, with no --install-dir/--model-path: the DEFAULT (per-user) shape."""
    return install.build_arg_parser().parse_args(argv or [])


def _fake_cli_home_manager(monkeypatch, *, state: ServiceState = ServiceState.RUNNING):
    """The service manager the per-user branch reports through, patched on cli_home (where register
    imports it) so one patch reaches the registration and the status below it."""
    mgr = MagicMock()
    mgr.status.return_value = ServiceStatus(state, detail="fake")
    factory = MagicMock(return_value=mgr)
    monkeypatch.setattr(cli_home, "get_service_manager", factory)
    return factory, mgr


@pytest.fixture
def per_user(monkeypatch, tmp_path):
    """Everything the default install touches outside the network/subprocess boundaries mocked:
    the per-user Layout, this "checkout" (a fake source tree), the shim's bin dir and HOME, and
    detect_os. Returns (layout, source_root)."""
    lay = _layout(tmp_path)
    source = _fake_source(tmp_path)
    monkeypatch.setattr(
        install.layout, "get_layout",
        lambda os_name, system_scope=False, override_root=None: lay,
    )
    monkeypatch.setattr(install, "REPO_ROOT", source)
    monkeypatch.setattr(install, "detect_os", lambda: "linux")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cli_shim, "_bin_dir", lambda profile, system_scope: tmp_path / "bin")
    monkeypatch.setattr(cli_shim, "_login_shell", lambda: "bash")
    monkeypatch.setattr(cli_shim.Path, "home", classmethod(lambda cls: home))
    return lay, source


def _installed(lay: Layout) -> InstalledCli:
    inst = cli_home.read_installed(cli_home.cli_home(lay))
    assert inst is not None, "nothing staged in the CLI home"
    return inst


class TestRunInstallPerUser:
    """The DEFAULT install registers the service and the CLI shim against the shared CLI home
    (`<install root>/cli`), not against whatever tree ran the install.

    Why it matters: that tree may be a GUI bundle or a checkout the person deletes tomorrow. Before
    the CLI home existed, whichever installer ran last owned the service and the `aipotluck-local-
    client` command, and it could quietly be the older of the two -- llama.cpp and newt pins
    included.
    """

    def test_the_service_runs_the_staged_copy_not_this_checkout(self, tmp_path, fake_fetch, per_user, monkeypatch):
        lay, source = per_user
        _, mgr = _fake_cli_home_manager(monkeypatch, state=ServiceState.RUNNING)

        rc = install.run_install(_parse())

        assert rc == 0
        inst = _installed(lay)
        _, kwargs = mgr.install.call_args
        assert kwargs["exec_path"] == inst.python
        assert kwargs["args"][0] == str(inst.src / "aipotluck" / "service" / "aipotluck_service.py")
        assert kwargs["working_dir"] == inst.src
        assert kwargs["system_scope"] is False
        assert kwargs["args"][1:] == ["--config-dir", str(lay.config_dir), "--log-dir", str(lay.log_dir)]
        assert str(source) not in str(kwargs["args"][0])
        # A running service is restarted onto the new copy rather than left on the old one.
        mgr.stop.assert_called_once_with(cli_home.SERVICE_NAME)
        mgr.start.assert_called_once_with(cli_home.SERVICE_NAME, system_scope=False)

    def test_the_staged_script_really_lives_under_the_cli_home(self, tmp_path, fake_fetch, per_user, monkeypatch):
        # The path assertions above would also pass for a staged tree that landed somewhere else;
        # this pins the location to <install root>/cli/src-<sha12>.
        lay, _ = per_user
        _fake_cli_home_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)

        install.run_install(_parse())

        inst = _installed(lay)
        home = lay.install_root / cli_home.CLI_HOME_DIRNAME
        assert inst.src == home / f"src-{SHA_NEW[:12]}"
        assert (inst.src / "aipotluck" / "service" / "aipotluck_service.py").is_file()
        assert (home / cli_home.MANIFEST_NAME).is_file()

    def test_the_shim_runs_the_cli_from_the_cli_home(self, tmp_path, fake_fetch, per_user, monkeypatch):
        lay, source = per_user
        _fake_cli_home_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)

        install.run_install(_parse())

        inst = _installed(lay)
        text = (tmp_path / "bin" / cli_shim.CLI_SHIM_NAME).read_text(encoding="utf-8")
        cli_py = inst.src / "aipotluck" / "installer" / "cli.py"
        assert f'exec "{inst.python}" "{cli_py}" "$@"' in text
        assert str(source) not in text

    def test_bundled_python_dir_and_installed_by_reach_adopt(self, tmp_path, fake_fetch, per_user, monkeypatch):
        lay, _ = per_user
        _fake_cli_home_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)
        real_adopt = cli_home.adopt
        calls = []

        def _spy(source_root, lay, **kwargs):
            calls.append((source_root, lay, kwargs))
            return real_adopt(source_root, lay, **kwargs)

        monkeypatch.setattr(cli_home, "adopt", _spy)
        bundle = tmp_path / "bundle" / "python"

        rc = install.run_install(_parse(["--bundled-python-dir", str(bundle), "--installed-by", "gui"]))

        assert rc == 0
        assert len(calls) == 1
        assert calls[0][0] == install.REPO_ROOT
        assert calls[0][1] == lay
        assert calls[0][2] == {"bundled_python_dir": bundle, "installed_by": "gui"}
        assert _installed(lay).installed_by == "gui"

    def test_superseded_staged_copies_are_pruned(self, tmp_path, fake_fetch, per_user, monkeypatch):
        lay, _ = per_user
        stale = cli_home.cli_home(lay) / ("src-" + "0" * 12)
        stale.mkdir(parents=True)
        _fake_cli_home_manager(monkeypatch, state=ServiceState.NOT_INSTALLED)

        install.run_install(_parse())

        inst = _installed(lay)
        assert not stale.exists()
        assert inst.src.is_dir()

    def test_the_summary_names_the_cli_home_copy(self, tmp_path, fake_fetch, per_user, monkeypatch, capsys):
        lay, _ = per_user
        _fake_cli_home_manager(monkeypatch, state=ServiceState.RUNNING)

        install.run_install(_parse())

        out = capsys.readouterr().out
        assert f"CLI home:       {_installed(lay).src}" in out

    def test_no_service_stages_nothing_into_the_cli_home(self, tmp_path, fake_fetch, per_user, monkeypatch, capsys):
        # `--no-service` stays exactly what it was: fetch/extract llama.cpp, write runtime.json,
        # stop. No CLI home, no shim, no service manager.
        lay, _ = per_user
        factory, mgr = _fake_cli_home_manager(monkeypatch)
        adopt = MagicMock()
        monkeypatch.setattr(cli_home, "adopt", adopt)
        monkeypatch.setattr(install, "get_service_manager", MagicMock())

        rc = install.run_install(_parse(["--no-service"]))

        assert rc == 0
        adopt.assert_not_called()
        factory.assert_not_called()
        mgr.install.assert_not_called()
        assert not (lay.install_root / cli_home.CLI_HOME_DIRNAME).exists()
        assert not (tmp_path / "bin").exists()
        out = capsys.readouterr().out
        assert "CLI home:" not in out
        assert "not installed (--no-service)" in out


class TestInstallDirKeepsThisCheckout:
    """--install-dir and --system are untouched by the CLI home: no staging, no manifest, and the
    service still runs this checkout out of the root the user named."""

    def test_install_dir_never_consults_or_creates_a_cli_home(
        self, tmp_path, fake_fetch, monkeypatch, _never_a_real_os_service
    ):
        cli_home_factory, _ = _never_a_real_os_service
        adopt = MagicMock()
        monkeypatch.setattr(cli_home, "adopt", adopt)
        mgr = MagicMock()
        mgr.status.return_value = ServiceStatus(ServiceState.RUNNING, detail="active")
        monkeypatch.setattr(install, "get_service_manager", lambda os_name: mgr)
        monkeypatch.setattr(install, "ensure_python", lambda: Path("/usr/bin/python3"))
        monkeypatch.setattr(install, "install_cli_shim", lambda *a, **kw: Path("/fake/bin/aipotluck-local-client"))
        install_dir = tmp_path / "install"

        rc = install.run_install(make_args(install_dir))

        assert rc == 0
        adopt.assert_not_called()
        cli_home_factory.assert_not_called()
        assert not (install_dir / cli_home.CLI_HOME_DIRNAME).exists()
        _, kwargs = mgr.install.call_args
        assert kwargs["exec_path"] == Path("/usr/bin/python3")
        assert kwargs["args"][0] == str(install.REPO_ROOT / "aipotluck" / "service" / "aipotluck_service.py")
        assert kwargs["working_dir"] == install.REPO_ROOT


class TestDelegationToANewerCliHome:
    """An older copy of this installer (a GUI bundle from last month, a stale checkout) must hand
    the WHOLE install to the newer copy in the CLI home, not just decline to replace its tree:
    otherwise llama.cpp and newt would be re-pinned to older versions while the CLI stayed new.
    """

    def _plant_installed(self, lay: Layout, *, sha: str, commit_time: int) -> InstalledCli:
        """An already-installed CLI home, exactly as a previous run would have left it."""
        home = cli_home.cli_home(lay)
        src = home / f"src-{sha[:12]}"
        (src / "aipotluck" / "installer").mkdir(parents=True)
        (src / "aipotluck" / "installer" / "install.py").write_text("# newer install\n", encoding="utf-8")
        python = src / "python3"
        python.write_text("", encoding="utf-8")
        inst = InstalledCli(BuildInfo(sha, commit_time), src, python, "cli", "2026-01-01T00:00:00Z")
        cli_home.write_installed(home, inst)
        return inst

    def _setup(self, tmp_path, monkeypatch, *, installed: tuple[str, int] | None, source: tuple[str, int]):
        lay = _layout(tmp_path)
        monkeypatch.setattr(
            install.layout, "get_layout", lambda os_name, system_scope=False, override_root=None: lay
        )
        monkeypatch.setattr(install, "detect_os", lambda: "linux")
        monkeypatch.setattr(install, "REPO_ROOT", _fake_source(tmp_path, sha=source[0], commit_time=source[1]))
        inst = None
        if installed is not None:
            inst = self._plant_installed(lay, sha=installed[0], commit_time=installed[1])
        return lay, inst

    def test_a_newer_installed_copy_receives_the_whole_install(self, tmp_path, monkeypatch):
        lay, inst = self._setup(
            tmp_path, monkeypatch, installed=(SHA_NEW, TIME_NEW), source=(SHA_OLD, TIME_OLD)
        )
        calls = []

        def _execve(path, argv, env):
            calls.append((path, argv, env))
            raise SystemExit(0)  # stands in for the process being replaced

        monkeypatch.setattr(install.os, "execve", _execve)
        run_install = MagicMock(return_value=0)
        monkeypatch.setattr(install, "run_install", run_install)

        with pytest.raises(SystemExit):
            install.main(["--no-start", "--model-path", "./m.gguf"])

        assert len(calls) == 1
        path, argv, env = calls[0]
        assert path == str(inst.python)
        assert argv == [
            str(inst.python),
            str(inst.src / "aipotluck" / "installer" / "install.py"),
            "--no-start",
            "--model-path",
            "./m.gguf",
        ]
        assert env[install.DELEGATED_ENV] == "1"
        # The marker goes to the child only -- this process's own environment is left alone.
        assert install.DELEGATED_ENV not in os.environ
        run_install.assert_not_called()

    def test_windows_runs_the_newer_copy_and_returns_its_exit_code(self, tmp_path, monkeypatch):
        _, inst = self._setup(
            tmp_path, monkeypatch, installed=(SHA_NEW, TIME_NEW), source=(SHA_OLD, TIME_OLD)
        )
        # os.name cannot be patched on the real module (pathlib would start handing out
        # WindowsPath), so install.py's view of `os` is replaced with a proxy that only lies about
        # name -- the branch under test is os.name == "nt", and Windows has no execve.
        class _FakeOs:
            name = "nt"

            def __init__(self, real):
                self._real = real

            def __getattr__(self, item):
                return getattr(self._real, item)

        monkeypatch.setattr(install, "os", _FakeOs(os))
        call = MagicMock(return_value=7)
        monkeypatch.setattr(install.subprocess, "call", call)
        run_install = MagicMock(return_value=0)
        monkeypatch.setattr(install, "run_install", run_install)

        assert install.main(["--no-start"]) == 7

        argv, kwargs = call.call_args
        assert argv[0] == [
            str(inst.python),
            str(inst.src / "aipotluck" / "installer" / "install.py"),
            "--no-start",
        ]
        assert kwargs["env"][install.DELEGATED_ENV] == "1"
        run_install.assert_not_called()

    def _assert_no_delegation(self, monkeypatch, tmp_path, argv, *, installed, source):
        self._setup(tmp_path, monkeypatch, installed=installed, source=source)
        execve = MagicMock()
        monkeypatch.setattr(install.os, "execve", execve)
        args = install.build_arg_parser().parse_args(argv)
        assert install._maybe_delegate(args, argv) is None
        execve.assert_not_called()

    def test_a_delegated_run_installs_itself_instead_of_handing_over_again(self, tmp_path, monkeypatch):
        # Without this the newer copy -- which is older than nothing, so it would delegate back --
        # would hand straight back to the caller and the two would ping-pong forever.
        monkeypatch.setenv(install.DELEGATED_ENV, "1")
        self._assert_no_delegation(
            monkeypatch, tmp_path, [], installed=(SHA_NEW, TIME_NEW), source=(SHA_OLD, TIME_OLD)
        )

    def test_an_identical_installed_build_is_not_a_reason_to_delegate(self, tmp_path, monkeypatch):
        self._assert_no_delegation(
            monkeypatch, tmp_path, [], installed=(SHA_NEW, TIME_NEW), source=(SHA_NEW, TIME_NEW)
        )

    def test_an_older_installed_build_is_not_a_reason_to_delegate(self, tmp_path, monkeypatch):
        self._assert_no_delegation(
            monkeypatch, tmp_path, [], installed=(SHA_OLD, TIME_OLD), source=(SHA_NEW, TIME_NEW)
        )

    def test_nothing_installed_is_not_a_reason_to_delegate(self, tmp_path, monkeypatch):
        self._assert_no_delegation(
            monkeypatch, tmp_path, [], installed=None, source=(SHA_NEW, TIME_NEW)
        )

    def test_system_installs_never_delegate(self, tmp_path, monkeypatch):
        # --system installs machine-wide into /opt/aipotluck: there is no per-user CLI home to hand
        # over to, and the shim it writes belongs to no user either.
        self._assert_no_delegation(
            monkeypatch, tmp_path, ["--system"], installed=(SHA_NEW, TIME_NEW), source=(SHA_OLD, TIME_OLD)
        )

    def test_install_dir_installs_never_delegate(self, tmp_path, monkeypatch):
        self._assert_no_delegation(
            monkeypatch, tmp_path, ["--install-dir", str(tmp_path / "elsewhere")],
            installed=(SHA_NEW, TIME_NEW), source=(SHA_OLD, TIME_OLD),
        )
