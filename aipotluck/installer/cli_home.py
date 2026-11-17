"""The CLI home: one copy of this source tree per user, owned by neither installer.

Why it exists: `aipotluck-install` and the desktop GUI each used to keep their own copy of the
tree, so a machine could end up with two CLIs at different versions, and whichever one wrote the
`aipotluck-local-client` shim and the OS service last silently won -- including downgrading to an
older copy. Instead both installers stage what they have into a single neutral folder,
`<layout.install_root>/cli` (the per-user layout only; `--system`/`--install-dir` installs keep
their old behaviour), and the shim and service always point there. Newer wins: a copy is only
replaced when its source commit is *strictly* newer than the installed one, so the order the two
installers happen to run in stops mattering, and uninstalling the GUI cannot take the CLI the
CLI installer put there with it.

Version = the source commit, never a hand-maintained version string. `aipotluck/_build_info.txt`
holds `$Format:%H %ct$` and `.gitattributes` marks it `export-subst`, so anything built with
`git archive` (the GUI's bundled tree, a GitHub tarball) ships the real `<sha> <unix time>`; in a
plain git clone the file stays unsubstituted and `read_build_info` falls back to `git log -1`,
but only when that root really is a checkout (never a repo further up the tree -- a dotfiles repo
in $HOME must never be mistaken for this project's history). Neither source -> sha "unknown",
time 0, which loses to anything already installed.

`current.json` (schema 1) is the manifest: which source directory and interpreter the staged copy
uses, who staged it, and when. This module owns reading and writing both, atomically (write a
temp file, then `os.replace`), so a half-written manifest or build-info file is never observed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from aipotluck.installer import layout
from aipotluck.proc import NO_WINDOW

log = logging.getLogger("aipotluck.installer.cli_home")

CLI_HOME_DIRNAME = "cli"
MANIFEST_NAME = "current.json"
BUILD_INFO_RELPATH = Path("aipotluck") / "_build_info.txt"
SCHEMA = 1

UNKNOWN_SHA = "unknown"
_GIT_TIMEOUT_S = 10
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class BuildInfo:
    sha: str
    commit_time: int

    def short(self) -> str:
        return self.sha[:12]


@dataclass(frozen=True)
class InstalledCli:
    build: BuildInfo
    src: Path
    python: Path
    installed_by: str
    installed_at: str


def _parse_build_info(text: str) -> BuildInfo | None:
    """`<40-hex sha> <unix seconds>`, or None for anything else -- including the unsubstituted
    `$Format:%H %ct$` a git clone's working tree contains."""
    if text.lstrip().startswith("$Format"):
        return None
    parts = text.split()
    if len(parts) != 2:
        return None
    sha, when = parts
    if not _SHA_RE.match(sha) or not when.isdigit():
        return None
    return BuildInfo(sha, int(when))


def _git_build_info(source_root: Path) -> BuildInfo | None:
    """The commit this checkout is on, or None. Only ever asked about source_root itself: a `.git`
    higher up the tree is a different project (a dotfiles repo in $HOME, say) and must be ignored,
    which is why the caller checks for source_root/.git before this runs."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "log", "-1", "--format=%H %ct"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S, **NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _parse_build_info(result.stdout.strip())


def read_build_info(source_root: Path) -> BuildInfo:
    """Version of the tree at source_root: the substituted `_build_info.txt` if it has one, else
    the checkout's own HEAD, else BuildInfo("unknown", 0). Never raises -- an unknown version is a
    real state (an unpacked copy of a copy), not an error, and it simply never beats an installed
    one."""
    source_root = Path(source_root)
    try:
        text = (source_root / BUILD_INFO_RELPATH).read_text(encoding="utf-8")
        build = _parse_build_info(text)
    except (OSError, UnicodeDecodeError):
        build = None
    if build is not None:
        return build
    if (source_root / ".git").exists():
        fallback = _git_build_info(source_root)
        if fallback is not None:
            return fallback
    return BuildInfo(UNKNOWN_SHA, 0)


def is_newer(candidate: BuildInfo, installed: BuildInfo) -> bool:
    """Strictly newer, by source commit time -- so the same commit installed by the other
    installer never counts as an upgrade, and no install can downgrade."""
    return candidate.commit_time > installed.commit_time


def cli_home(lay: layout.Layout) -> Path:
    return lay.install_root / CLI_HOME_DIRNAME


def read_installed(home: Path) -> InstalledCli | None:
    """What is staged in `home`, or None if nothing usable is: no manifest, unreadable/dead JSON,
    an unknown schema, a missing or mistyped key, or a manifest whose referenced source tree or
    interpreter has since been deleted. Never raises -- None means "nothing to keep", which every
    caller treats as "install fresh"."""
    try:
        data = json.loads((Path(home) / MANIFEST_NAME).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            return None
        if isinstance(data.get("schema"), bool):
            return None
        sha, commit_time = data.get("sha"), data.get("commit_time")
        src, python = data.get("src"), data.get("python")
        installed_by, installed_at = data.get("installed_by"), data.get("installed_at")
        if not isinstance(commit_time, int) or isinstance(commit_time, bool):
            return None
        if not isinstance(sha, str) or not isinstance(src, str) or not isinstance(python, str):
            return None
        if not isinstance(installed_by, str) or not isinstance(installed_at, str):
            return None
        if not (sha and src and python and installed_by and installed_at):
            return None
        src_path, python_path = Path(src), Path(python)
        if not src_path.is_dir() or not python_path.is_file():
            return None
        return InstalledCli(BuildInfo(sha, commit_time), src_path, python_path, installed_by, installed_at)
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None


def write_installed(home: Path, inst: InstalledCli) -> Path:
    """Writes `home/current.json` and returns it. Atomic: a temp file in the same directory then
    `os.replace`, so a crash mid-write leaves the previous manifest intact rather than a truncated
    one -- an interrupted install must not look like a corrupt CLI home."""
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    target = home / MANIFEST_NAME
    tmp = home / f"{MANIFEST_NAME}.tmp-{os.getpid()}"
    payload = {
        "schema": SCHEMA,
        "sha": inst.build.sha,
        "commit_time": inst.build.commit_time,
        "src": str(inst.src),
        "python": str(inst.python),
        "installed_by": inst.installed_by,
        "installed_at": inst.installed_at,
    }
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return target
