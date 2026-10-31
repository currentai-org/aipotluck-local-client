"""Cross-platform Python interpreter detection + best-effort auto-install.

Why this exists: the installer itself is a Python script, so whoever
invokes `python -m aipotluck.installer.install` already has *some* Python. But the
services we register (systemd/launchd/Scheduled Task/Windows Service) all
re-invoke `sys.executable` later, at boot/login, potentially on a machine
where Python was only ever present via a venv that's since been removed,
or (most commonly) a fresh Windows machine with **no system Python at
all** reached via our packaging/windows/install.ps1 bootstrapper rather
than a manual `python -m aipotluck.installer.install` call.

This module is the shared logic behind that bootstrapper:
  - find_python(): locate a suitable interpreter already on the system
  - ensure_python(): find one, or (Windows only) try to install one via
    winget, then re-resolve

macOS/Linux ship python3 in practice on every currently-supported OS
release, so ensure_python() on those platforms only detects and raises a
clear, actionable error -- it does not attempt sudo-gated installs.
Windows is the one platform where "no Python at all" is the common case
for a non-developer end user, so it's the one platform with a real
auto-install path, per explicit product requirement.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("aipotluck.installer.python_bootstrap")

MIN_PYTHON = (3, 9)
WINGET_PACKAGE_ID = "Python.Python.3.12"


class PythonNotFoundError(RuntimeError):
    pass


def _version_ok(exe: str) -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            [exe, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        major, minor = (int(p) for p in result.stdout.strip().split("."))
    except ValueError:
        return None
    version = (major, minor)
    return version if version >= MIN_PYTHON else None


def _candidate_names() -> list[str]:
    if sys.platform == "win32":
        # 'py' is the Windows launcher and is the most reliable way to find
        # *a* Python even when python.exe isn't directly on PATH.
        return ["py", "python", "python3"]
    return ["python3", "python"]


def find_python() -> Path | None:
    """Return a path to a suitable (>=3.9) Python interpreter, preferring
    the currently-running interpreter if it already qualifies."""
    current = (sys.version_info.major, sys.version_info.minor)
    if current >= MIN_PYTHON:
        return Path(sys.executable)

    for name in _candidate_names():
        exe = shutil.which(name)
        if exe and _version_ok(exe):
            return Path(exe)
    return None


def _windows_common_install_dirs() -> list[Path]:
    local_appdata = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = []
    for base in (local_appdata / "Programs" / "Python", program_files):
        if base.exists():
            candidates.extend(sorted(base.glob("Python3*"), reverse=True))
    return candidates


def _try_winget_install() -> bool:
    if shutil.which("winget") is None:
        log.warning("winget not found on PATH; cannot auto-install Python")
        return False
    cmd = [
        "winget", "install", "--id", WINGET_PACKAGE_ID, "-e",
        "--silent", "--accept-package-agreements", "--accept-source-agreements",
    ]
    log.info("Installing Python via winget: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.error("winget install failed to run: %s", exc)
        return False
    if result.returncode != 0:
        log.error("winget install exited %s: %s", result.returncode, result.stderr.strip())
        return False
    log.info("winget reports Python installed successfully")
    return True


def ensure_python(auto_install: bool = True) -> Path:
    """Find a suitable Python, installing one if possible and necessary.
    Raises PythonNotFoundError with an actionable message if none can be
    found or installed.

    STUB STATUS (Windows auto-install path): implemented against documented
    winget CLI behavior and never run -- every Windows machine tested so far
    already had a suitable Python, so this branch was not reached.
    """
    found = find_python()
    if found:
        return found

    if sys.platform == "win32" and auto_install:
        log.warning("No suitable Python (>=%d.%d) found; attempting winget install", *MIN_PYTHON)
        if _try_winget_install():
            # winget updates the machine/user PATH registry keys, but this
            # already-running process won't see that until it restarts --
            # so re-resolve via shutil.which() first (works if winget
            # updated the *process* environment via a broadcasted env
            # change, which it usually does not), then fall back to
            # scanning well-known install directories directly.
            found = find_python()
            if found:
                return found
            for candidate_dir in _windows_common_install_dirs():
                candidate = candidate_dir / "python.exe"
                if candidate.exists() and _version_ok(str(candidate)):
                    return candidate
            raise PythonNotFoundError(
                "winget reported a successful Python install, but this process "
                "can't see it yet (PATH not refreshed in the current session). "
                "Please close and re-open your terminal / re-run the installer."
            )
        raise PythonNotFoundError(
            "No Python installation found and winget is unavailable or failed. "
            "Install Python 3.9+ manually from https://www.python.org/downloads/windows/ "
            "(check 'Add python.exe to PATH' during install), then re-run the installer."
        )

    if sys.platform == "darwin":
        raise PythonNotFoundError(
            "No Python 3.9+ found. Install it with 'brew install python3' "
            "(see https://brew.sh) or from https://www.python.org/downloads/macos/, "
            "then re-run the installer."
        )

    raise PythonNotFoundError(
        "No Python 3.9+ found. Install it with your distro's package manager "
        "(e.g. 'sudo apt install python3' / 'sudo dnf install python3'), "
        "then re-run the installer."
    )
