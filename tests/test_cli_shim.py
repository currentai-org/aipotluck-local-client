"""aipotluck.installer.cli_shim -- writes the `aipotluck-local-client` wrapper onto PATH.

install_cli_shim() itself is only ever exercised against a tmp_path-redirected bin dir (via
monkeypatching `_bin_dir`) -- never the real `~/.local/bin` or `/usr/local/bin`, so running this
suite never touches or requires write access to the actual host.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from aipotluck.installer import cli_shim
from aipotluck.installer.platform_detect import HostProfile


class TestBinDir:
    def test_linux_user_scope_is_local_bin(self):
        profile = HostProfile(os_name="linux", arch="x64", backend="cpu")
        assert cli_shim._bin_dir(profile, system_scope=False) == Path.home() / ".local" / "bin"

    def test_linux_system_scope_is_usr_local_bin(self):
        profile = HostProfile(os_name="linux", arch="x64", backend="cpu")
        assert cli_shim._bin_dir(profile, system_scope=True) == Path("/usr/local/bin")

    def test_macos_matches_linux(self):
        profile = HostProfile(os_name="macos", arch="arm64", backend="cpu")
        assert cli_shim._bin_dir(profile, system_scope=False) == Path.home() / ".local" / "bin"
        assert cli_shim._bin_dir(profile, system_scope=True) == Path("/usr/local/bin")

    def test_windows_user_scope_uses_localappdata(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
        profile = HostProfile(os_name="windows", arch="x64", backend="cpu")
        assert cli_shim._bin_dir(profile, system_scope=False) == tmp_path / "AppData" / "Local" / "aipotluck" / "bin"

    def test_windows_system_scope_uses_program_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
        profile = HostProfile(os_name="windows", arch="x64", backend="cpu")
        assert cli_shim._bin_dir(profile, system_scope=True) == tmp_path / "Program Files" / "aipotluck" / "bin"


class TestShimText:
    def test_posix_shim_execs_the_given_python_and_cli_module(self):
        text = cli_shim._posix_shim_text(Path("/usr/bin/python3"))
        assert text.startswith("#!/usr/bin/env bash\n")
        assert f'exec "/usr/bin/python3" "{cli_shim.CLI_MODULE_PATH}" "$@"' in text

    def test_windows_shim_calls_the_given_python_and_cli_module(self):
        text = cli_shim._windows_shim_text(Path(r"C:\Python312\python.exe"))
        assert text.startswith("@echo off\r\n")
        assert f'"C:\\Python312\\python.exe" "{cli_shim.CLI_MODULE_PATH}" %*' in text
        assert "\r\n" in text  # real Windows line endings, not bare \n


class TestShellProfiles:
    """The PATH entry for new shells: a marked block in the user's startup files. Real files under
    tmp_path, and real shells where the host has them -- the block has to actually work."""

    def test_writes_the_login_shells_file_when_there_are_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli_shim, "_login_shell", lambda: "zsh")
        written = cli_shim.add_to_shell_profiles(tmp_path / "bin", home=tmp_path)
        assert written == [tmp_path / ".zshenv"]  # macOS's default: zsh, no dotfiles yet

    def test_adds_to_every_existing_startup_file_and_keeps_their_contents(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli_shim, "_login_shell", lambda: "bash")
        (tmp_path / ".bashrc").write_text("alias ll='ls -l'\n")
        (tmp_path / ".profile").write_text("export EDITOR=vi")  # no trailing newline
        written = cli_shim.add_to_shell_profiles(tmp_path / "bin", home=tmp_path)
        assert sorted(written) == sorted([tmp_path / ".bashrc", tmp_path / ".profile"])
        assert (tmp_path / ".bashrc").read_text().startswith("alias ll='ls -l'\n")
        assert (tmp_path / ".profile").read_text().startswith("export EDITOR=vi\n")

    def test_is_idempotent_and_a_new_bin_dir_replaces_the_old_one(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli_shim, "_login_shell", lambda: "bash")
        cli_shim.add_to_shell_profiles(tmp_path / "old", home=tmp_path)
        cli_shim.add_to_shell_profiles(tmp_path / "new", home=tmp_path)
        cli_shim.add_to_shell_profiles(tmp_path / "new", home=tmp_path)
        text = (tmp_path / ".profile").read_text()
        assert text.count(cli_shim.PROFILE_BEGIN) == 1
        assert str(tmp_path / "new") in text and str(tmp_path / "old") not in text

    def test_remove_restores_the_file_exactly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli_shim, "_login_shell", lambda: "bash")
        original = "# my profile\nexport EDITOR=vi\n"
        (tmp_path / ".profile").write_text(original)
        cli_shim.add_to_shell_profiles(tmp_path / "bin", home=tmp_path)
        assert cli_shim.remove_from_shell_profiles(home=tmp_path) == [tmp_path / ".profile"]
        assert (tmp_path / ".profile").read_text() == original

    def test_remove_leaves_files_without_the_block_untouched(self, tmp_path):
        (tmp_path / ".bashrc").write_text("x\n")
        assert cli_shim.remove_from_shell_profiles(home=tmp_path) == []
        assert (tmp_path / ".bashrc").read_text() == "x\n"

    @pytest.mark.parametrize("shell, startup", [("bash", ".bashrc"), ("bash", ".profile"), ("zsh", ".zshenv"), ("sh", ".profile")])
    def test_a_real_new_shell_finds_the_command(self, monkeypatch, tmp_path, shell, startup):
        exe = shutil.which(shell)
        if exe is None:
            pytest.skip(f"no {shell} on this host")
        monkeypatch.setattr(cli_shim, "_login_shell", lambda: shell)
        (tmp_path / startup).touch()
        bin_dir = tmp_path / "bin dir"  # a space, as in a macOS "Application Support" path
        bin_dir.mkdir()
        tool = bin_dir / "aipotluck-local-client"
        tool.write_text("#!/bin/sh\necho found-it\n")
        tool.chmod(0o755)
        cli_shim.add_to_shell_profiles(bin_dir, home=tmp_path)
        env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "ENV": str(tmp_path / ".profile")}
        flags = {".bashrc": ["-i", "-c"], ".profile": ["-l", "-c"], ".zshenv": ["-c"]}[startup]
        if shell == "sh":
            flags = ["-i", "-c"]  # POSIX sh reads $ENV for interactive shells
        out = subprocess.run([exe, *flags, "aipotluck-local-client"], env=env, capture_output=True, text=True, timeout=20)
        assert "found-it" in out.stdout, (out.stdout, out.stderr)


class TestInstallCliShimPosix:
    @pytest.fixture(autouse=True)
    def _no_real_profiles(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli_shim.Path, "home", classmethod(lambda cls: tmp_path / "home"))
        (tmp_path / "home").mkdir(exist_ok=True)

    def test_user_install_puts_the_bin_dir_on_path_for_new_shells(self, monkeypatch, tmp_path):
        bin_dir = self._redirect_bin_dir(monkeypatch, tmp_path)
        monkeypatch.setattr(cli_shim, "_login_shell", lambda: "bash")
        cli_shim.install_cli_shim(HostProfile(os_name="linux", arch="x64", backend="cpu"), Path("/usr/bin/python3"), system_scope=False)
        assert str(bin_dir) in (tmp_path / "home" / ".profile").read_text()

    def test_system_install_leaves_shell_profiles_alone(self, monkeypatch, tmp_path):
        self._redirect_bin_dir(monkeypatch, tmp_path)
        cli_shim.install_cli_shim(HostProfile(os_name="linux", arch="x64", backend="cpu"), Path("/usr/bin/python3"), system_scope=True)
        assert list((tmp_path / "home").iterdir()) == []

    def _redirect_bin_dir(self, monkeypatch, tmp_path):
        target = tmp_path / "bin"
        monkeypatch.setattr(cli_shim, "_bin_dir", lambda profile, system_scope: target)
        return target

    def test_writes_executable_file_with_correct_content(self, monkeypatch, tmp_path):
        bin_dir = self._redirect_bin_dir(monkeypatch, tmp_path)
        profile = HostProfile(os_name="linux", arch="x64", backend="cpu")
        python_exe = Path("/usr/bin/python3")

        shim_path = cli_shim.install_cli_shim(profile, python_exe, system_scope=False)

        assert shim_path == bin_dir / cli_shim.CLI_SHIM_NAME
        assert shim_path.exists()
        content = shim_path.read_text()
        assert str(python_exe) in content
        assert str(cli_shim.CLI_MODULE_PATH) in content

        mode = shim_path.stat().st_mode
        assert mode & stat.S_IXUSR
        assert mode & stat.S_IXGRP
        assert mode & stat.S_IXOTH

    def test_is_idempotent_and_fully_overwrites(self, monkeypatch, tmp_path):
        bin_dir = self._redirect_bin_dir(monkeypatch, tmp_path)
        profile = HostProfile(os_name="linux", arch="x64", backend="cpu")

        cli_shim.install_cli_shim(profile, Path("/usr/bin/python3.9"), system_scope=False)
        shim_path = cli_shim.install_cli_shim(profile, Path("/usr/bin/python3.12"), system_scope=False)

        content = shim_path.read_text()
        assert "/usr/bin/python3.12" in content
        assert "/usr/bin/python3.9" not in content

    def test_creates_bin_dir_if_missing(self, monkeypatch, tmp_path):
        bin_dir = tmp_path / "does" / "not" / "exist" / "yet"
        monkeypatch.setattr(cli_shim, "_bin_dir", lambda profile, system_scope: bin_dir)
        profile = HostProfile(os_name="linux", arch="x64", backend="cpu")

        cli_shim.install_cli_shim(profile, Path("/usr/bin/python3"), system_scope=False)

        assert bin_dir.is_dir()


class TestInstallCliShimWindows:
    @pytest.fixture(autouse=True)
    def _never_the_real_user_path(self, monkeypatch, request):
        """On a Windows host the real _add_to_user_path_windows writes HKCU\\Environment\\Path --
        found doing exactly that, leaving pytest temp dirs on the test account's PATH. Only the
        test that is about winreg being absent may reach the real function, and only off Windows."""
        if request.node.name == "test_gracefully_handles_no_winreg_available":
            if os.name == "nt":
                pytest.skip("would write the real user PATH; the no-winreg branch only exists off Windows")
            return
        monkeypatch.setattr(cli_shim, "_add_to_user_path_windows", lambda bin_dir: True)

    def _redirect_bin_dir(self, monkeypatch, tmp_path):
        target = tmp_path / "bin"
        monkeypatch.setattr(cli_shim, "_bin_dir", lambda profile, system_scope: target)
        return target

    def test_writes_cmd_file_with_crlf(self, monkeypatch, tmp_path):
        bin_dir = self._redirect_bin_dir(monkeypatch, tmp_path)
        profile = HostProfile(os_name="windows", arch="x64", backend="cpu")

        shim_path = cli_shim.install_cli_shim(profile, Path(r"C:\Python\python.exe"), system_scope=False)

        assert shim_path == bin_dir / f"{cli_shim.CLI_SHIM_NAME}.cmd"
        raw = shim_path.read_bytes()
        assert b"\r\n" in raw

    def test_system_scope_never_attempts_a_registry_write(self, monkeypatch, tmp_path, caplog):
        # System-scope PATH edits need elevation this process may not have -- install_cli_shim must
        # print the manual instructions rather than attempt (and fail/crash on) a registry write.
        self._redirect_bin_dir(monkeypatch, tmp_path)
        called = []
        monkeypatch.setattr(cli_shim, "_add_to_user_path_windows", lambda bin_dir: called.append(bin_dir) or True)
        profile = HostProfile(os_name="windows", arch="x64", backend="cpu")

        with caplog.at_level(logging.WARNING, logger="aipotluck.installer.cli_shim"):
            cli_shim.install_cli_shim(profile, Path(r"C:\Python\python.exe"), system_scope=True)

        assert called == []
        assert any("machine-wide PATH" in rec.message for rec in caplog.records)

    def test_user_scope_calls_add_to_user_path(self, monkeypatch, tmp_path):
        self._redirect_bin_dir(monkeypatch, tmp_path)
        called = []
        monkeypatch.setattr(cli_shim, "_add_to_user_path_windows", lambda bin_dir: called.append(bin_dir) or True)
        profile = HostProfile(os_name="windows", arch="x64", backend="cpu")

        cli_shim.install_cli_shim(profile, Path(r"C:\Python\python.exe"), system_scope=False)

        assert len(called) == 1

    def test_gracefully_handles_no_winreg_available(self, monkeypatch, tmp_path, caplog):
        # This suite runs on Linux, so the real `_add_to_user_path_windows` hits its own
        # `except ImportError` branch for real (no mocking) -- proving the Windows shim path
        # doesn't crash when winreg genuinely isn't importable, not just when it's mocked away.
        self._redirect_bin_dir(monkeypatch, tmp_path)
        profile = HostProfile(os_name="windows", arch="x64", backend="cpu")

        with caplog.at_level(logging.WARNING, logger="aipotluck.installer.cli_shim"):
            shim_path = cli_shim.install_cli_shim(profile, Path(r"C:\Python\python.exe"), system_scope=False)

        assert shim_path.exists()
        assert any("could not be added automatically" in rec.message for rec in caplog.records)
