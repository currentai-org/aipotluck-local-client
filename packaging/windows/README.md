# Windows

## Entry point: `install.ps1`

`install.ps1` is the real Windows entry point -- unlike Linux/macOS, a
"just run `python -m aipotluck.installer.install`" instruction is unsafe to hand to a
typical Windows user because there may be **no Python installed at all**.
This script is plain PowerShell (no dependencies) and:

1. Looks for a working Python (`py`/`python`/`python3`) >= 3.9 on PATH.
2. If none found, installs Python via `winget install Python.Python.3.12`
   (silent, auto-accept agreements).
3. Re-resolves the interpreter path (winget doesn't refresh PATH in the
   current session, so this probes `%LOCALAPPDATA%\Programs\Python\` and
   `%ProgramFiles%` directly if `winget`'s own install can't yet be seen).
4. Hands off to `python -m aipotluck.installer.install` with equivalent flags
   (`-System`, `-Backend`, `-ModelHf`, `-Tag`, `-NoStart`, `-Verbose`).

Usage:
```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\install.ps1
powershell -ExecutionPolicy Bypass -File packaging\windows\install.ps1 -System -Backend cuda
```

The same find-or-install-Python logic also lives in
`aipotluck/installer/python_bootstrap.py` (`ensure_python()`), which the Python
installer itself calls before registering any service -- so even if
someone invokes `python -m aipotluck.installer.install` directly on a box where
Python was found via some now-stale path, the service registration step
re-validates and, on Windows, can still trigger the same winget install.

## Service backends

- Default (no admin): a Scheduled Task (`schtasks /sc onlogon`), started at
  user logon. See `aipotluck/installer/service/windows_service.py`.
- `--system` (requires elevation): a real Windows Service registered via
  `pywin32`, implemented in `aipotluck/service/windows_service_host.py`
  (`win32serviceutil.ServiceFramework` subclass wrapping the same
  `AipotluckServiceRunner` used by every other OS). Started at boot, no
  login required, restarted by the Windows SCM on crash.

`pywin32` is only required for the `--system` path; declared as the
`windows` extra in `pyproject.toml` (`pip install
aipotluck-local-client[windows]`).

## Status

Tested on Windows 10 22H2: the per-user logon task registers as a standard
user, reports its real state, and the service it starts tears its children
down on any kind of stop (the Job Object). Everything downstream of
"find/install a working Python interpreter and hand off to
`aipotluck/installer/install.py`" is the exact same code path already tested
end-to-end on Linux and run on Windows (`aipotluck/installer/`, `aipotluck/service/`).

Unverified here, narrowly: winget's actual install behavior (the test
machine already had Python, and a machine without it is the case that
matters), and the pywin32 ServiceFramework contract used by `--system`.

## Future work

A double-click MSI/EXE wrapper around `install.ps1` (e.g. via Inno Setup
or a signed PS1-to-EXE bundler) so end users don't need to open PowerShell
manually. Not implemented yet.
