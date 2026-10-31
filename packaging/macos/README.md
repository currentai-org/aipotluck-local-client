# macOS

## Entry point

Same as Linux: `python3 -m aipotluck.installer.install`. macOS ships `python3` by
default on every currently-supported release, so there's no
find-or-install-Python bootstrap step here (unlike Windows) -- see
`aipotluck/installer/python_bootstrap.py::ensure_python()`, which on macOS just
raises a clear "brew install python3" message if somehow no Python 3.9+ is
found (e.g. a stripped-down CI image).

## Service backend: launchd

Implemented in `aipotluck/installer/service/launchd.py`, invoking the exact same
`aipotluck/service/aipotluck_service.py` script as Linux/Windows (no macOS-specific
service code beyond the plist itself -- full portability of the actual
supervision logic, see `aipotluck/service/runner.py`).

- Default (user/agent scope): plist at
  `~/Library/LaunchAgents/com.aipotluck.<name>.plist`, registered with
  `launchctl bootstrap`/`kickstart` in the user's domain (and
  `launchctl enable` first, since a label left disabled refuses to
  bootstrap). Starts at login, `KeepAlive: true` restarts it on crash.
- `--system` scope: `/Library/LaunchDaemons/`, requires sudo, starts at
  boot without login.

## Status

Tested on real macOS hardware: a Mac mini on macOS 27 and an M1 MacBook Air.
Install, `status`, `login`, `logout` and `uninstall` were all run there --
which is how the two bugs this backend now avoids surfaced (`KeepAlive`
restarting a job that was asked to stop, and the legacy `unload -w` leaving a
permanent `disable` override that made every later reinstall fail). The
service payload it launches (`aipotluck/service/aipotluck_service.py` ->
`aipotluck/service/runner.py` -> `aipotluck/service/llama_supervisor.py`) is
the identical, already-tested-on-Linux code.

## Future work

A .pkg installer + notarization workflow around
`python -m aipotluck.installer.install`. Not implemented yet.
