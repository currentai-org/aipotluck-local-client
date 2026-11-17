#!/usr/bin/env python3
"""Adopt this source tree into the per-user CLI home and re-point the service and shim at it.

Usage:
    python -m aipotluck.installer.adopt [options]

This is the entry point the desktop GUI runs on every launch: "here is the tree I was shipped with
-- make it the installed CLI unless a newer one is already installed". What it does, in order:

    1. Refuse to do anything at all under --if-installed when this user has never installed
       anything (no runtime.json): a GUI opened on a machine without a service started must not
       create a CLI home, a shim, or anything else behind the person's back.
    2. Delegate the version decision to cli_home.adopt -- adopt, "current", or keep the newer one.
    3. Re-point the OS service (only one that is already registered -- never create one where none
       exists, which is the GUI-hosted/no-systemd case) and the `aipotluck-local-client` shim at
       the staged copy's interpreter and source tree, whenever something was staged or the shim is
       aimed anywhere else.
    4. Prune older staged copies.

THE OUTPUT CONTRACT: the last line on stdout is `ADOPT_RESULT <json>`, one of

    {"action": "adopted", "sha": ..., "installed_sha": ..., "repointed": bool,
     "service_restarted": bool, "python": ..., "src": ...}
    {"action": "current" | "kept-newer", ...same fields...}
    {"action": "skipped", "reason": "not installed"}
    {"action": "error", "error": "<message>"}          (exit code 1)

and nothing else is ever printed to stdout -- log output goes to stderr -- so the GUI can parse
that line rather than scraping human prose. `sha` is the commit this tree is, `installed_sha` the
one the CLI home now holds (they differ for "kept-newer").
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aipotluck.installer import cli_home, cli_shim, layout
from aipotluck.installer.platform_detect import HostProfile, detect_arch, detect_os
from aipotluck.installer.service.base import ServiceState

log = logging.getLogger("aipotluck.installer.adopt")

RESULT_PREFIX = "ADOPT_RESULT "
RUNTIME_CONFIG_NAME = "runtime.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aipotluck-adopt",
        description="Stage this copy of the CLI into the per-user CLI home and re-point the "
                    "service and the aipotluck-local-client shim at it.",
    )
    parser.add_argument(
        "--bundled-python-dir", type=Path, default=None,
        help="The directory holding the interpreter running this code, when it is a bundled one "
             "(the GUI's): copied into the CLI home so the installed CLI and service keep working "
             "if that bundle goes away. Omitted, the running interpreter's own path is used.",
    )
    parser.add_argument(
        "--installed-by", choices=["cli", "gui"], default="cli",
        help="Recorded in the CLI home's manifest, so `status` can say which installer staged the "
             "copy. Default: cli.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Stage this tree even when the CLI home already holds the same or a newer commit "
             "(a deliberate reinstall, or a corrupt home).",
    )
    parser.add_argument(
        "--if-installed", action="store_true",
        help="Do nothing (print action 'skipped') unless this user has installed before, i.e. "
             "unless runtime.json exists. What the GUI passes on every launch.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def emit(payload: dict) -> None:
    """The one line the GUI reads. Always the last thing on stdout."""
    print(RESULT_PREFIX + json.dumps(payload), flush=True)


def run(args: argparse.Namespace) -> int:
    profile = HostProfile(os_name=detect_os(), arch=detect_arch(), backend="cpu")
    lay = layout.get_layout(profile.os_name)

    if args.if_installed and not (lay.config_dir / RUNTIME_CONFIG_NAME).exists():
        log.info("Nothing installed for this user yet (%s missing); nothing to adopt.", lay.config_dir / RUNTIME_CONFIG_NAME)
        emit({"action": "skipped", "reason": "not installed"})
        return 0

    candidate = cli_home.read_build_info(REPO_ROOT)
    result = cli_home.adopt(
        REPO_ROOT,
        lay,
        bundled_python_dir=args.bundled_python_dir,
        installed_by=args.installed_by,
        force=args.force,
    )
    inst = result.installed
    log.info("CLI home build: %s (this tree: %s) at %s", inst.build.short(), candidate.short(), inst.src)

    # Re-pointing is only worth doing when something changed underneath: either this tree was just
    # staged, or the shim is aimed at another interpreter/tree (a pre-CLI-home install, a moved
    # checkout, a hand edit).
    repoint = result.action == "adopted" or not cli_shim.shim_matches(profile, inst.python, inst.src)
    service_restarted = False
    if repoint:
        # cli_home's own seam, so this check and the registration it decides both go through the
        # same manager (and the same monkeypatch point in tests).
        mgr = cli_home.get_service_manager(profile.os_name)
        service_present = mgr.status(cli_home.SERVICE_NAME).state is not ServiceState.NOT_INSTALLED
        _, service_restarted, shim_path = cli_home.register(
            profile, lay, inst, register_service=service_present, start=False
        )
        log.info("Pointed the CLI shim at %s (service %s)", shim_path, "re-pointed" if service_present else "not installed")

    home = cli_home.cli_home(lay)
    removed = cli_home.prune(home, cli_home.keep_set(home, inst))
    if removed:
        log.info("Removed %d superseded staged cop%s from %s", len(removed), "y" if len(removed) == 1 else "ies", home)

    emit(
        {
            "action": result.action,
            "sha": candidate.sha,
            "installed_sha": inst.build.sha,
            "repointed": repoint,
            "service_restarted": service_restarted,
            "python": str(inst.python),
            "src": str(inst.src),
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        return run(args)
    except Exception as exc:  # top-level guard: the GUI gets a parseable line, not a traceback
        log.error("Adopt failed: %s", exc, exc_info=args.verbose)
        emit({"action": "error", "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
