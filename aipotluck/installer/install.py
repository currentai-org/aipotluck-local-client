#!/usr/bin/env python3
"""aipotluck-local-client installer.

Usage:
    python -m aipotluck.installer.install [options]

Flow:
    1. Detect platform/arch/GPU-backend.
    2. Decide whether a prebuilt llama.cpp release asset is actually viable for this host, or
       whether it needs to be built from source instead (build_strategy.py) -- e.g. every
       Jetson-class arm64+CUDA board: nvidia-smi is real there, but no linux-arm64-cuda asset is
       ever published upstream, and even the linux-arm64-cpu asset needs a newer glibc than
       JetPack ships. See build_strategy.py's module docstring for the confirmed evidence.
    3a. Prebuilt (upstream) path: download (cached, checksum-verified) + extract it.
    3b. Prebuilt (our own cache) path: when no upstream asset is viable, check our own
        llama_custom_builds.json for a binary already built for this exact host/backend/GPU
        architecture combination (build_cache.py) -- skips compiling entirely on a repeat match.
    3c. Source-build path: only when neither of the above matches. Check the host actually has
        what a build needs; for the apt-fixable gaps (cmake/git/compiler/OpenSSL headers -- never
        a GPU vendor toolchain), offer to install them via sudo with explicit consent
        (--allow-apt-install, or an interactive y/N prompt -- never unconditionally), then
        configure + build llama-server from vendor/llama.cpp (source_build.py).
    4. Write runtime.json pointing at the resolved llama-server binary.
    5. Register our blank Python service against this user's CLI home -- `<install root>/cli`, the
       one copy of this tree the CLI and GUI installers share (cli_home.py) -- rather than against
       whatever checkout happened to run the install.
    6. Install the `aipotluck-local-client` CLI shim onto PATH, running that same staged copy.
       `--system` and `--install-dir` installs keep their old shape: service and shim both run this
       checkout directly.
    7. Optionally start the service (default: yes).

Before any of that, a per-user run hands itself over to a NEWER copy already staged in the CLI home
(`_maybe_delegate`): the whole install -- llama.cpp and newt pins included -- must never move a
machine backwards just because an older installer (a GUI bundle, a stale checkout) ran last. The
delegated run is marked through AIPOTLUCK_INSTALL_DELEGATED so it can never bounce again.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aipotluck.installer import (
    build_cache,
    build_strategy,
    cli_home,
    fetch,
    layout,
    model_pull,
    model_sizing,
    python_bootstrap,
    source_build,
)
from aipotluck.installer.platform_detect import HostProfile, detect_host_profile, detect_os
from aipotluck.installer.python_bootstrap import PythonNotFoundError, ensure_python
from aipotluck.installer.service.base import ServiceState, get_service_manager
from aipotluck.installer.cli_shim import CLI_SHIM_NAME, install_cli_shim
from aipotluck.service.runner import DEFAULT_HOST, DEFAULT_MODELS_MAX, DEFAULT_PORT

log = logging.getLogger("aipotluck.installer")

SERVICE_NAME = "aipotluck"
# Set on the environment of a delegated run so it installs itself instead of handing over again.
DELEGATED_ENV = "AIPOTLUCK_INSTALL_DELEGATED"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8080
DEFAULT_MODEL_HF = "bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
PRESETS_FILENAME = "llama_presets.ini"  # lives in the per-OS config dir, alongside runtime.json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aipotluck-installer",
        description="Install llama.cpp + the aipotluck local service.",
    )
    parser.add_argument("--tag", default=None, help="Override the pinned llama.cpp release tag")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cpu", "cuda", "vulkan", "rocm", "metal"],
        help="GPU backend to use (default: auto-detect, falls back to cpu; Apple-silicon Macs are metal)",
    )
    parser.add_argument(
        "--install-dir", type=Path, default=None,
        help="Override the default per-OS install root",
    )
    parser.add_argument(
        "--system", action="store_true",
        help="Install machine-wide (requires elevation); default is per-user, no admin",
    )
    parser.add_argument(
        "--bundled-python-dir", type=Path, default=None,
        help="The directory holding the interpreter running this installer, when that interpreter "
             "is a bundled one (the GUI's): copied into the CLI home, so the installed CLI and "
             "service keep working if that bundle is later moved or removed. Per-user installs "
             "only; omitted, the running interpreter's own path is used as is (a system python is "
             "already durable).",
    )
    parser.add_argument(
        "--installed-by", choices=["cli", "gui"], default="cli",
        help="Recorded in the CLI home's manifest so `status` can say which installer staged the "
             "copy -- and which one to uninstall first. Default: cli.",
    )
    parser.add_argument(
        "--no-start", action="store_true",
        help="Install the service but do not start it",
    )
    parser.add_argument(
        "--no-service", action="store_true",
        help="Only fetch/extract llama.cpp; skip installing the Python service "
             "(useful for CI / dry runs on a host with no service manager)",
    )
    parser.add_argument(
        "--via", default="direct", choices=["direct"],
        help="Install method. Only 'direct' (our own download+verify) is "
             "implemented; 'brew'/'winget' passthroughs are future work.",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-hf", default=DEFAULT_MODEL_HF,
        help="Hugging Face repo[:quant] to pre-fetch and pre-size as the device's default model, "
             "e.g. 'bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M' "
             f"(default: {DEFAULT_MODEL_HF} -- a small placeholder model so there's something "
             "ready out of the box; override for real use). llama-server itself runs in router "
             "mode and can serve any model already downloaded, not only this one -- "
             "see `pull` for downloading/sizing more later.",
    )
    model_group.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to a local GGUF file to pre-size as the device's default model, instead of "
             "fetching one from Hugging Face. Its containing directory is exposed to the router "
             "via --models-dir, since a loose local file isn't visible to the router's own "
             "HF-cache auto-discovery.",
    )
    parser.add_argument(
        "--no-model-pull", action="store_true",
        help="Skip pre-fetching/pre-sizing --model-hf at install time -- useful in CI/dry-run "
             "contexts that don't want the network/subprocess cost. The device still ends up "
             "with a working router; its first-ever model load just uses llama-server's own "
             "stock defaults until you run `pull` (or restart the service once something's "
             "cached, which backfills a preset for it -- see runner.py's own startup step).",
    )
    parser.add_argument(
        "--models-max", type=int, default=DEFAULT_MODELS_MAX,
        help=f"How many models llama-server's router may hold loaded at once (default: "
             f"{DEFAULT_MODELS_MAX} -- keeps aipotluck.installer.model_sizing's memory budgeting, "
             "which assumes one model at a time, valid; raising this is a real capacity tradeoff, "
             "not a free win)",
    )
    parser.add_argument(
        "--gpu-layers", default="auto",
        help="llama-server -ngl value: an integer, 'auto', or 'all' (default: auto). Global -- "
             "applies to every model the router loads, not just the default one (see "
             "runner.build_llama_server_args' own docstring for why this one stays router-level).",
    )
    parser.add_argument("--server-host", default=DEFAULT_SERVER_HOST, help="llama-server bind host")
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT, help="llama-server bind port")

    parser.add_argument(
        "--no-source-build", action="store_true",
        help="Fail instead of building llama-server from source when no prebuilt release asset is "
             "viable for this host (e.g. Jetson-class arm64+CUDA boards, or an old-glibc Linux host) "
             "-- useful in CI/dry-run contexts that want to know about the gap rather than sit "
             "through a build that can take well over an hour",
    )
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="Parallel build jobs for a from-source build (default: auto, capped by available RAM "
             "-- a naive -j$(nproc) can OOM a low-memory board like a Jetson Orin Nano)",
    )
    parser.add_argument(
        "--build-timeout", type=float, default=source_build.DEFAULT_BUILD_TIMEOUT_SECONDS,
        help=f"Wall-clock ceiling in seconds for a from-source build (default: "
             f"{source_build.DEFAULT_BUILD_TIMEOUT_SECONDS:.0f})",
    )
    apt_install_group = parser.add_mutually_exclusive_group()
    apt_install_group.add_argument(
        "--allow-apt-install", action="store_true",
        help="Consent up front to installing missing from-source-build dependencies (cmake, git, "
             "a C++ compiler, OpenSSL headers) via 'sudo apt-get install' -- skips the interactive "
             "y/N prompt. sudo still prompts for a password on its own terminal as usual; this "
             "installer never supplies one. The CUDA toolkit (nvcc) is never auto-installed "
             "regardless of this flag -- see source_build.missing_apt_packages' docstring.",
    )
    apt_install_group.add_argument(
        "--no-apt-install", action="store_true",
        help="Never offer to auto-install build dependencies, even interactively -- always just "
             "print the missing packages and the exact apt command, and stop.",
    )

    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _confirm_apt_install(packages: list[str]) -> bool:
    """Interactive y/N consent to auto-installing `packages` via sudo. Never prompts (comes back
    False) without a real controlling terminal on stdin -- a piped install (e.g. the public
    `curl ... | bash` one-liner) has no stdin a person could answer through, so the safe default
    there is "don't act without --allow-apt-install", not a prompt nobody can see or answer."""
    if not sys.stdin.isatty():
        return False
    try:
        answer = input(
            f"Install missing build dependencies now via sudo apt-get ({', '.join(packages)})? [y/N]: "
        ).strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def run_install(args: argparse.Namespace) -> int:
    profile: HostProfile = detect_host_profile(args.backend)
    log.info("Detected host profile: os=%s arch=%s backend=%s", profile.os_name, profile.arch, profile.backend)

    lay = layout.get_layout(profile.os_name, system_scope=args.system, override_root=args.install_dir)
    layout.ensure_layout_dirs(lay)
    log.info("Install root: %s", lay.install_root)
    log.info("Config dir:   %s", lay.config_dir)
    log.info("Log dir:      %s", lay.log_dir)

    manifest_path = REPO_ROOT / "llama_version.json"
    manifest = fetch.load_version_manifest(manifest_path)
    if args.tag:
        manifest["tag"] = args.tag
        log.info("Overriding pinned tag -> %s", args.tag)

    strategy = build_strategy.resolve_install_strategy(profile, manifest)
    used_source_build = False
    if strategy.use_source_build:
        custom_manifest_path = REPO_ROOT / "llama_custom_builds.json"
        custom_manifest = build_cache.load_custom_manifest(custom_manifest_path)
        custom_match = build_cache.find_custom_build(profile, custom_manifest, cuda_arch=strategy.cuda_arch)

        if custom_match is not None:
            custom_key, custom_entry = custom_match
            log.info(
                "No viable upstream asset for this host (%s), but a cached custom build exists "
                "(%s) -- skipping compilation", strategy.reason, custom_key,
            )
            archive_path = fetch.download_asset(custom_manifest, custom_entry, lay.state_dir / "downloads")
            llama_dir = fetch.extract_archive(
                archive_path, lay.install_root / "llama.cpp", custom_manifest["tag"], custom_key
            )
            server_bin = fetch.find_binary(llama_dir, "llama-server")
            asset_key = f"custom-build:{custom_key}"
        else:
            log.info(
                "No viable prebuilt asset for this host (%s) -- building llama-server from source",
                strategy.reason,
            )
            if args.no_source_build:
                log.error(
                    "--no-source-build set; refusing to build (%s). Drop that flag to let the "
                    "installer build llama-server from source instead.", strategy.reason,
                )
                return 1
            if profile.backend == "cuda" and strategy.cuda_arch is None:
                log.error(
                    "CUDA backend requested/detected but the GPU's compute capability could not be "
                    "determined (checked `nvidia-smi --query-gpu=compute_cap`) -- can't safely pick a "
                    "CMAKE_CUDA_ARCHITECTURES value. Check nvidia-smi works, or pass --backend cpu."
                )
                return 1

            problems = source_build.check_build_prerequisites(backend=profile.backend)
            if problems:
                apt_packages = source_build.missing_apt_packages() if not args.no_apt_install else []
                if apt_packages and source_build.has_apt() and (
                    args.allow_apt_install or _confirm_apt_install(apt_packages)
                ):
                    try:
                        source_build.install_apt_packages(apt_packages)
                    except source_build.AptInstallError as exc:
                        log.error("Installing dependencies failed: %s", exc)
                        return 1
                    problems = source_build.check_build_prerequisites(backend=profile.backend)

            if problems:
                log.error("Can't build llama-server from source -- missing prerequisites:")
                for problem in problems:
                    log.error("  - %s", problem)
                log.error("Install the above, then re-run this installer.")
                return 1

            free_gb = source_build.check_free_disk_gb(lay.install_root)
            if free_gb is not None and free_gb < source_build.MIN_FREE_DISK_GB:
                log.error(
                    "Only %.1fGB free at %s; a from-source build needs roughly %dGB. Free up space "
                    "and re-run.", free_gb, lay.install_root, source_build.MIN_FREE_DISK_GB,
                )
                return 1

            build_root = lay.install_root / "llama.cpp-build"
            log.info(
                "Building llama-server (tag=%s, cuda_arch=%s) -- this can take well over an hour on "
                "a low-power board; output follows live", manifest["tag"], strategy.cuda_arch,
            )
            server_bin = source_build.ensure_llama_server_built(
                REPO_ROOT, build_root, manifest["tag"],
                cuda_arch=strategy.cuda_arch, jobs=args.jobs, timeout=args.build_timeout,
            )
            asset_key = f"source-build:{profile.asset_key}"
            llama_dir = server_bin.parent.parent
            used_source_build = True
    else:
        asset_key, asset_entry = strategy.asset_key, strategy.asset_entry
        log.info("Resolved asset: %s (%s)", asset_key, asset_entry["file"])

        archive_path = fetch.download_asset(manifest, asset_entry, lay.state_dir / "downloads")
        llama_dir = fetch.extract_archive(archive_path, lay.install_root / "llama.cpp", manifest["tag"], asset_key)
        fetch.fetch_companions(manifest, asset_entry, lay.state_dir / "downloads", llama_dir)

        server_bin = fetch.find_binary(llama_dir, "llama-server")

    log.info("llama-server binary: %s", server_bin)

    presets_path = lay.config_dir / PRESETS_FILENAME
    runtime_config = {
        "llama_cpp": {
            "tag": manifest["tag"],
            "asset_key": asset_key,
            "built_from_source": used_source_build,
            "install_dir": str(llama_dir),
            "server_binary": str(server_bin),
            "lib_dir": str(server_bin.parent),
            "host": args.server_host,
            "port": args.server_port,
            "gpu_layers": args.gpu_layers,
            "presets_path": str(presets_path),
            "models_dir": str(args.model_path.parent) if args.model_path else None,
            "models_max": args.models_max,
        },
        "service": {
            "name": SERVICE_NAME,
            "config_dir": str(lay.config_dir),
            "log_dir": str(lay.log_dir),
        },
        # A FRESH install starts logged out -- no Pangolin credentials, so the service (runner.py)
        # holds both llama-server and newt back until `aipotluck-local-client login` sets this
        # true. This is what lets the public one-line installer (install.sh) take zero arguments:
        # it never needs a tunnel id/secret/endpoint in hand to finish installing. An install over
        # an ALREADY-PAIRED device keeps its pairing -- see _carry_over_pairing below.
        "logged_in": False,
    }
    _carry_over_pairing(lay.config_dir / "runtime.json", runtime_config)

    # Pre-fetch (HF only -- a --model-path file is already local) and pre-size the default model,
    # so a fresh install already has a correctly-sized router preset waiting rather than falling
    # back to llama-server's stock defaults until the person runs `pull` themselves. Never fatal to
    # the install -- the router itself works fine with zero presets, it just serves that one model
    # unsized until backfilled (see runner.backfill_missing_presets).
    if not args.no_model_pull:
        model_hf = args.model_hf if args.model_path is None else None
        model_id = model_hf if model_hf else args.model_path.name.replace(".gguf", "")
        try:
            if model_hf:
                log.info("Pre-fetching the default model (%s) -- this can take a while for a large quant", model_hf)
                model_pull.pull_model(server_bin, model_hf, on_progress=model_pull.terminal_progress())
            log.info("Sizing the default model (%s)", model_id)
            model_sizing.ensure_preset(
                server_bin, presets_path, model_id,
                model_hf=model_hf, model_path=args.model_path, gpu_layers=args.gpu_layers, force=True,
            )
        except (model_pull.ModelPullError, model_sizing.ModelSizingError) as exc:
            log.warning(
                "Could not pre-fetch/size the default model (%s) -- it'll run with llama-server's "
                "own defaults until you run `%s pull ...` or restart the service once it's cached.",
                exc, CLI_SHIM_NAME,
            )

    runtime_path = lay.config_dir / "runtime.json"
    runtime_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
    log.info("Wrote runtime config: %s", runtime_path)

    if args.no_service:
        log.info("--no-service set; skipping Python service install and the CLI PATH shim.")
        _print_summary(profile, lay, runtime_config, service_status=None, shim_path=None)
        return 0

    shim_path: Path | None = None
    cli_src: Path | None = None

    if args.system or args.install_dir is not None:
        # --system and --install-dir installs keep their old shape: the service and the shim run
        # this checkout, because there is no per-user CLI home involved at all (a raised install
        # root has no user to share it with, and /opt/aipotluck is not where a per-user shim's
        # cli.py lives either).
        service_mgr = get_service_manager(profile.os_name)
        service_script = REPO_ROOT / "aipotluck" / "service" / "aipotluck_service.py"
        service_args = [
            "--config-dir", str(lay.config_dir),
            "--log-dir", str(lay.log_dir),
        ]
        try:
            python_exe = ensure_python()
        except PythonNotFoundError as exc:
            log.error("%s", exc)
            raise
        log.info("Using Python interpreter for service: %s", python_exe)
        service_installed = True
        try:
            service_mgr.install(
                SERVICE_NAME,
                exec_path=python_exe,
                args=[str(service_script)] + service_args,
                system_scope=args.system,
                working_dir=REPO_ROOT,
                log_dir=lay.log_dir,
            )
            log.info("Installed %s service (system_scope=%s)", SERVICE_NAME, args.system)
        except RuntimeError as exc:
            # Everything else (llama.cpp, the model, runtime.json, the CLI) is in place and usable;
            # only the OS service is missing, and the summary below says so. Logged as an error, not
            # dropped: before this, a failed Windows registration was silently reported as installed.
            service_installed = False
            log.error("%s", exc)

        shim_path = install_cli_shim(profile, python_exe, args.system)

        if service_installed and not args.no_start:
            try:
                service_mgr.start(SERVICE_NAME, system_scope=args.system)
                log.info("Started %s service", SERVICE_NAME)
            except RuntimeError as exc:
                # Registered but not running now (Windows: the logon task can't be started from a
                # session with no desktop). It will start at the next sign-in; the summary below
                # shows its real state rather than a "Started" that didn't happen.
                log.warning("%s", exc)
    else:
        # Per-user: the CLI home is the one copy of this tree that both installers stage into, and
        # the service and shim are pointed at it -- never at this checkout, which may be a GUI
        # bundle or a working tree that is about to be deleted.
        if sys.version_info < python_bootstrap.MIN_PYTHON:
            log.error(
                "The per-user CLI home needs Python %d.%d or newer to run the staged copy; this "
                "installer is running on %d.%d. Upgrade Python and re-run (or use --install-dir "
                "for a self-contained install of this tree).",
                *python_bootstrap.MIN_PYTHON, *sys.version_info[:2],
            )
            return 1
        result = cli_home.adopt(
            REPO_ROOT, lay,
            bundled_python_dir=args.bundled_python_dir,
            installed_by=args.installed_by,
        )
        inst = result.installed
        cli_src = inst.src
        log.info(
            "The service and the %s command will run the CLI-home copy (%s): %s (build %s)",
            CLI_SHIM_NAME, result.action, inst.src, inst.build.short(),
        )
        service_installed, _restarted, shim_path = cli_home.register(
            profile, lay, inst, register_service=True, start=not args.no_start
        )
        if service_installed:
            log.info("Installed %s service (per-user, from the CLI home)", SERVICE_NAME)
        home = cli_home.cli_home(lay)
        removed = cli_home.prune(home, cli_home.keep_set(home, inst))
        if removed:
            log.info(
                "Removed %d superseded staged cop%s from %s",
                len(removed), "y" if len(removed) == 1 else "ies", home,
            )
        # Same seam cli_home.register reports through, so the status below describes the unit that
        # was just registered rather than a second, separately-constructed manager.
        service_mgr = cli_home.get_service_manager(profile.os_name)

    status = service_mgr.status(SERVICE_NAME, system_scope=args.system)
    _print_summary(
        profile, lay, runtime_config, service_status=status, shim_path=shim_path, cli_src=cli_src
    )

    if status.state not in (ServiceState.RUNNING, ServiceState.STOPPED):
        log.warning("Service status is ambiguous: %s (%s)", status.state, status.detail)

    return 0


def _carry_over_pairing(runtime_path: Path, runtime_config: dict) -> None:
    """Keeps an already-paired device paired across a re-install.

    The installer rebuilds runtime.json from scratch every run, which is right for the llama.cpp
    section -- those paths and that tag describe what was just installed. It is wrong for the
    pairing. `logged_in` and the `tunnel` section are the user's credentials, not install output,
    and rebuilding without them silently unpairs the device: the running service keeps serving from
    the config it already holds in memory, so nothing looks wrong until the next restart, which may
    be days later and will look unrelated. Observed on a Jetson that had been re-installed to pick
    up a newer build and went dark at its next restart.

    Upgrading is the common reason to re-run this installer, so it must not cost a re-pairing."""
    try:
        existing = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(existing, dict):
        return
    if existing.get("logged_in") and isinstance(existing.get("tunnel"), dict):
        runtime_config["logged_in"] = True
        runtime_config["tunnel"] = existing["tunnel"]
        log.info("Keeping this device's existing pairing (it was already logged in)")


def _print_summary(
    profile: HostProfile,
    lay: layout.Layout,
    runtime_config: dict,
    service_status,
    shim_path: Path | None,
    cli_src: Path | None = None,
) -> None:
    print()
    print("=" * 60)
    print("aipotluck-local-client install summary")
    print("=" * 60)
    print(f"Platform:       {profile.os_name}/{profile.arch} (backend={profile.backend})")
    print(f"llama.cpp:      {runtime_config['llama_cpp']['tag']} [{runtime_config['llama_cpp']['asset_key']}]")
    print(f"  server binary: {runtime_config['llama_cpp']['server_binary']}")
    print(f"Install root:   {lay.install_root}")
    if cli_src is not None:
        # Only for the per-user install: this is the tree the service and the shim actually run,
        # and it is NOT the checkout that was unpacked to get here -- worth saying out loud.
        print(f"CLI home:       {cli_src}")
    print(f"Config:         {lay.config_dir / 'runtime.json'}")
    print(f"Logs:           {lay.log_dir}")
    if service_status is not None:
        print(f"Service:        {service_status.state.value} ({service_status.detail})")
        print(f"  service health:  http://{DEFAULT_HOST}:{DEFAULT_PORT}/healthz")
        print(f"  service status:  http://{DEFAULT_HOST}:{DEFAULT_PORT}/status")
        llama = runtime_config["llama_cpp"]
        print(
            f"  llama-server (router): http://{llama['host']}:{llama['port']}/models "
            "(managed by the service, once logged in)"
        )
    else:
        print("Service:        not installed (--no-service)")
    if shim_path is not None:
        print(f"CLI:            {shim_path}")
    print("=" * 60)
    print("You're logged out -- llama-server and the tunnel are held back until you pair this")
    print("device. Get a tunnel id/secret/endpoint from Settings -> Local Inference -> Add a")
    print("managed server on aipotluck.org, then run:")
    print()
    if shim_path is not None:
        print(f"    {CLI_SHIM_NAME} login")
        print()
        print(f"(open a new terminal first if this is the first time {CLI_SHIM_NAME} has been installed)")
    else:
        print("    python3 -m aipotluck.installer.cli login")
    print("=" * 60)


def _maybe_delegate(args: argparse.Namespace, raw_argv: list[str]) -> int | None:
    """Hand this whole install over to a newer copy in the CLI home, or None to install this tree.

    Newer wins, never downgrade -- and "the whole install" is the point: re-running an older
    installer (a GUI bundle left over from last month, a stale checkout) must not move this
    machine's llama.cpp or newt pins backwards, and must not replace the copy the service and the
    shim already run. So rather than merely keeping the staged tree, the older installer re-execs
    the newer copy's own install.py with the same arguments and lets it be the installer.

    Not for `--system`/`--install-dir` (those install this tree into a root of the user's choosing,
    with no CLI home involved), and not for a run that is itself a delegated one -- that env flag is
    what stops an A->B->A ping-pong between two copies that each think the other is newer.
    """
    if args.system or args.install_dir is not None:
        return None
    if os.environ.get(DELEGATED_ENV) == "1":
        return None
    lay = layout.get_layout(detect_os())
    installed = cli_home.read_installed(cli_home.cli_home(lay))
    candidate = cli_home.read_build_info(REPO_ROOT)
    if installed is None or not cli_home.is_newer(installed.build, candidate):
        return None
    log.info(
        "A newer aipotluck-local-client (%s) is already installed; handing this install to it "
        "(this copy is %s)",
        installed.build.short(), candidate.short(),
    )
    env = dict(os.environ, **{DELEGATED_ENV: "1"})
    # install.py's own file path, not `-m`: a module invocation would resolve `aipotluck` against
    # the delegated copy's root anyway, but the file path also leaves the child's CWD alone, so a
    # relative path in argv (--model-path ./x.gguf) still means what the caller meant.
    cmd = [
        str(installed.python),
        str(installed.src / "aipotluck" / "installer" / "install.py"),
        *raw_argv,
    ]
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        # Windows has no execve; the child runs in turn and its exit code becomes ours.
        return subprocess.call(cmd, env=env)
    os.execve(cmd[0], cmd, env)
    return None  # not reached: a successful execve never returns


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)
    setup_logging(args.verbose)
    try:
        delegated = _maybe_delegate(args, raw_argv)
        if delegated is not None:
            return delegated
        return run_install(args)
    except Exception as exc:  # top-level guard: always report clearly
        log.error("Install failed: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
