#!/usr/bin/env python3
"""Thin management CLI for an already-installed aipotluck-local-client device.

The public one-line installer (install.sh -> aipotluck.installer.install) always produces a
"logged out" install: no Pangolin credentials, and the service (aipotluck/service/runner.py) holds
both llama-server and newt back until this CLI's `login` command supplies them. This file is the
only place that edits runtime.json's `tunnel` section and `logged_in` flag after install time -- it
never touches the llama.cpp install or the OS service registration itself, it only flips that state
and restarts the already-installed service so it picks the change up.

The installer also drops a small `aipotluck-local-client` wrapper onto PATH that runs this exact
file (see cli_shim.py) -- that's the command to actually type; `python -m aipotluck.installer.cli`
below is the fallback for a `--no-service` install (which skips the shim) or a checkout PATH
doesn't reach yet.

Usage:
    aipotluck-local-client login    # prompts for the pairing JSON (paste it from Settings)
    aipotluck-local-client login --credentials-file creds.json   # or read it from a file
    aipotluck-local-client logout   # unpairs; llama-server and the tunnel stop until you log in again
    aipotluck-local-client status   # asks the running service for its login/tunnel/llama-server state
    aipotluck-local-client pull <hf-repo[:quant]>   # download + size + speed-check a model
    aipotluck-local-client list     # list models already downloaded locally, with their grades
    aipotluck-local-client benchmark [model]        # re-measure speed without re-downloading

llama-server itself runs in router mode (CUR-1965) -- one always-running process that serves
whichever model an inference request's own "model" field names, loading/unloading instances on
demand (see vendor/llama.cpp/tools/server/server-models.cpp). There's no single "active model" for
this CLI to set anymore; `pull` just makes sure a model is downloaded and correctly sized
(aipotluck.installer.model_sizing) before it's ever requested.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aipotluck.installer import (  # noqa: E402
    layout,
    model_perf,
    model_perf_live,
    model_perf_store,
    model_presets,
    newt_fetch,
)
from aipotluck.installer.cli_shim import CLI_SHIM_NAME  # noqa: E402
from aipotluck.installer.model_pull import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    ModelPullError,
    list_cached_models,
    pull_model,
)
from aipotluck.installer.model_sizing import ModelSizingError, ensure_preset  # noqa: E402
from aipotluck.installer.platform_detect import HostProfile, detect_host_profile  # noqa: E402
from aipotluck.installer.service.base import get_service_manager  # noqa: E402
from aipotluck.service.runner import DEFAULT_HOST, DEFAULT_PORT  # noqa: E402

log = logging.getLogger("aipotluck.cli")

SERVICE_NAME = "aipotluck"


def _common_args_parser() -> argparse.ArgumentParser:
    """Shared flags, added as a `parents=[...]` base to EACH SUBCOMMAND only -- never to the top
    parser too. argparse's subparsers action re-parses the chosen subcommand's tokens into a fresh
    namespace and then unconditionally overwrites the outer one with it (`_SubParsersAction.
    __call__` passes `None`, not the existing namespace, to the subparser's own parse_known_args) --
    so a flag declared on BOTH the top parser and a subparser silently reverts to the subparser's
    own default the moment it's given before the subcommand name, with no error. Confirmed live:
    `aipotluck-local-client --install-dir X login` silently ignored X and wrote to the default
    install dir instead; `aipotluck-local-client login --install-dir X` (this design) fails loudly
    on an unrecognized argument if given in the wrong position rather than acting on the wrong
    directory. Keep these flags after the subcommand name; don't hoist them onto the top parser to
    "support both orders" -- that's exactly the change that reintroduces the silent bug."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--install-dir", type=Path, default=None,
        help="Override the default per-OS install root -- must match what the installer used",
    )
    common.add_argument(
        "--system", action="store_true",
        help="Target a machine-wide (--system) install rather than the default per-user one",
    )
    common.add_argument("-v", "--verbose", action="store_true")
    return common


def build_arg_parser() -> argparse.ArgumentParser:
    common = _common_args_parser()
    parser = argparse.ArgumentParser(
        prog="aipotluck-local-client",
        description="Log in/out and check status for an already-installed aipotluck-local-client device.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    login = sub.add_parser("login", parents=[common], help="Pair this device with a managed tunnel endpoint")
    login.add_argument(
        "--credentials-file", type=Path, default=None,
        help="Read the pairing JSON ({tunnelId, tunnelSecret, tunnelEndpoint}) from this file instead "
             "of pasting it -- the same JSON Settings -> Local Inference's Copy button copies. Mutually "
             "exclusive with --tunnel-id/--tunnel-secret/--tunnel-endpoint.",
    )
    login.add_argument(
        "--tunnel-id", default=None,
        help="Skip the prompt (scripting only); must be given with the other two --tunnel-* flags, "
             "and not with --credentials-file",
    )
    login.add_argument("--tunnel-secret", default=None, help="Skip the prompt (scripting only)")
    login.add_argument("--tunnel-endpoint", default=None, help="Skip the prompt (scripting only)")

    sub.add_parser(
        "logout", parents=[common],
        help="Unpair this device; llama-server and the tunnel stop until you log in again",
    )
    sub.add_parser("status", parents=[common], help="Query the running service's login/tunnel/llama-server state")

    pull = sub.add_parser(
        "pull", parents=[common],
        help="Download and size a Hugging Face model ahead of time (router mode serves any "
             "downloaded model on request, so this isn't 'activating' a single one)",
    )
    pull.add_argument(
        "model",
        help="Hugging Face repo[:quant], e.g. bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M "
             "-- passed straight through to llama-server's own -hf downloader",
    )
    pull.add_argument(
        "--timeout", type=float, default=None,
        help=f"Seconds to wait for the download+load to finish before giving up "
             f"(default: {int(DEFAULT_TIMEOUT_SECONDS)})",
    )
    pull.add_argument(
        "--force", action="store_true",
        help="Keep the model even if it benchmarks too slowly to finish a conversation turn",
    )
    pull.add_argument(
        "--skip-benchmark", action="store_true",
        help="Download and size only, without measuring speed (no grade is recorded)",
    )

    sub.add_parser(
        "list", parents=[common],
        help="List models already downloaded locally (via llama-server's own --cache-list)",
    )

    benchmark = sub.add_parser(
        "benchmark", parents=[common],
        help="Re-measure how fast a cached model runs on this device, without re-downloading it",
    )
    benchmark.add_argument(
        "model", nargs="?", default=None,
        help="Hugging Face repo[:quant] to measure; omit to measure every cached model",
    )

    return parser


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def _profile_and_layout(args: argparse.Namespace) -> tuple[HostProfile, layout.Layout]:
    profile = detect_host_profile()
    lay = layout.get_layout(profile.os_name, system_scope=args.system, override_root=args.install_dir)
    return profile, lay


def _runtime_path(lay: layout.Layout) -> Path:
    return lay.config_dir / "runtime.json"


def _load_runtime(runtime_path: Path) -> dict:
    if not runtime_path.exists():
        log.error(
            "No runtime.json found at %s -- run the installer first (see README.md's Quick start).",
            runtime_path,
        )
        raise SystemExit(1)
    try:
        return json.loads(runtime_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Failed to read %s: %s", runtime_path, exc)
        raise SystemExit(1) from exc


def _save_runtime(runtime_path: Path, runtime_config: dict) -> None:
    runtime_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")


def _restart_service(os_name: str, system_scope: bool) -> None:
    """Best-effort: a `--no-service`/dry-run install has nothing registered to restart, and that's
    not this CLI's problem to fix -- just tell the person plainly rather than raising."""
    try:
        service_mgr = get_service_manager(os_name)
        service_mgr.stop(SERVICE_NAME, system_scope=system_scope)
        service_mgr.start(SERVICE_NAME, system_scope=system_scope)
        log.info("Restarted the %s service to pick up the change.", SERVICE_NAME)
    except Exception as exc:
        log.warning(
            "Could not restart the %s service (%s). If you're running it manually, restart it yourself.",
            SERVICE_NAME, exc,
        )


class CredentialsError(ValueError):
    pass


def _parse_credentials_json(raw: str) -> tuple[str, str, str]:
    """Parses the `{tunnelId, tunnelSecret, tunnelEndpoint}` JSON that Settings -> Local Inference's
    Copy button puts on the clipboard (compact, single line by design -- see that button's own
    comment) -- the same shape --credentials-file reads from disk. Raises CredentialsError with a
    message fit to show the user directly on anything wrong: bad JSON, wrong shape, missing/empty/
    non-string key."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CredentialsError(f"That's not valid JSON ({exc}).") from exc
    if not isinstance(data, dict):
        raise CredentialsError("Expected a JSON object with tunnelId/tunnelSecret/tunnelEndpoint.")

    required = ("tunnelId", "tunnelSecret", "tunnelEndpoint")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise CredentialsError(f"Missing (or empty) key(s): {', '.join(missing)}.")
    non_strings = [key for key in required if not isinstance(data[key], str)]
    if non_strings:
        raise CredentialsError(f"Key(s) must be strings: {', '.join(non_strings)}.")

    return data["tunnelId"], data["tunnelSecret"], data["tunnelEndpoint"]


def _read_hidden_line_with_feedback(prompt: str) -> str:
    """Like getpass.getpass, but echoes one '.' per character received instead of nothing at all.
    A silent prompt gave a paste no visible effect until Enter was pressed, which read as "did that
    even work?" -- the dots are just paste-landed feedback, not a strength meter, so a fixed
    placeholder character is fine even though it says nothing about length.

    Needs raw per-keystroke access to the terminal, which only a real interactive tty can give, so
    this falls back to plain getpass.getpass whenever stdin isn't one -- piped input, a redirected
    file, or (as in this module's own test suite) a monkeypatched getpass.getpass under pytest's
    captured, non-tty stdin.
    """
    if not sys.stdin.isatty():
        return getpass.getpass(prompt)

    print(prompt, end="", flush=True)
    try:
        chars = _read_hidden_chars_windows() if sys.platform == "win32" else _read_hidden_chars_unix()
    finally:
        print()  # move past the dots onto their own line, matching getpass's own trailing newline
    return "".join(chars)


def _read_hidden_chars_unix() -> list[str]:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    chars: list[str] = []
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n", ""):
                break
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\x7f", "\x08"):  # backspace/delete
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            chars.append(ch)
            sys.stdout.write(".")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return chars


def _read_hidden_chars_windows() -> list[str]:
    import msvcrt  # type: ignore[import-not-found]

    chars: list[str] = []
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            break
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x08":  # backspace
            if chars:
                chars.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        chars.append(ch)
        sys.stdout.write(".")
        sys.stdout.flush()
    return chars


def _prompt_credentials_json() -> tuple[str, str, str]:
    print("Pair this device with a managed tunnel endpoint.")
    print("Paste the JSON from Settings -> Local Inference -> Add a managed server (its Copy button")
    print("copies exactly this), then press Enter. Dots below confirm characters are being received.")
    print()
    while True:
        # Hidden, not plain input(): the pasted blob contains the tunnel secret, so it must not
        # echo to the terminal (or land in a screen recording/over-the-shoulder view) any more than
        # a bare secret prompt would have -- only the dot-per-character feedback is new.
        raw = _read_hidden_line_with_feedback("Credentials JSON: ").strip()
        if not raw:
            print("  (required, try again)")
            continue
        try:
            return _parse_credentials_json(raw)
        except CredentialsError as exc:
            print(f"  {exc} Try again.")


def run_login(args: argparse.Namespace) -> int:
    flags_given = [args.tunnel_id, args.tunnel_secret, args.tunnel_endpoint]
    if args.credentials_file and any(flags_given):
        log.error("Pass --credentials-file or --tunnel-id/--tunnel-secret/--tunnel-endpoint, not both.")
        return 1
    if any(flags_given) and not all(flags_given):
        log.error("Pass all three of --tunnel-id/--tunnel-secret/--tunnel-endpoint, or none to be prompted.")
        return 1

    if all(flags_given):
        tunnel_id, tunnel_secret, tunnel_endpoint = args.tunnel_id, args.tunnel_secret, args.tunnel_endpoint
    elif args.credentials_file:
        try:
            raw = args.credentials_file.read_text(encoding="utf-8")
        except OSError as exc:
            log.error("Could not read %s: %s", args.credentials_file, exc)
            return 1
        try:
            tunnel_id, tunnel_secret, tunnel_endpoint = _parse_credentials_json(raw)
        except CredentialsError as exc:
            log.error("%s (in %s)", exc, args.credentials_file)
            return 1
    else:
        tunnel_id, tunnel_secret, tunnel_endpoint = _prompt_credentials_json()

    profile, lay = _profile_and_layout(args)
    runtime_path = _runtime_path(lay)
    runtime_config = _load_runtime(runtime_path)

    newt_manifest = newt_fetch.load_newt_manifest(REPO_ROOT / "newt_version.json")
    newt_asset_key, newt_asset_entry = newt_fetch.resolve_newt_asset(newt_manifest, profile)
    log.info("Resolved newt asset: %s (%s)", newt_asset_key, newt_asset_entry["file"])
    newt_binary = newt_fetch.download_newt_binary(
        newt_manifest, newt_asset_entry, lay.install_root / "newt" / newt_manifest["tag"]
    )
    log.info("newt binary: %s", newt_binary)

    runtime_config["tunnel"] = {
        "provider": "pangolin",
        "binary": str(newt_binary),
        "id": tunnel_id,
        "secret": tunnel_secret,
        "endpoint": tunnel_endpoint,
    }
    runtime_config["logged_in"] = True
    _save_runtime(runtime_path, runtime_config)
    log.info("Wrote pairing to %s", runtime_path)

    _restart_service(profile.os_name, args.system)
    print()
    print("Logged in. llama-server and the tunnel are starting -- check with:")
    print(f"    {CLI_SHIM_NAME} status")
    return 0


def run_logout(args: argparse.Namespace) -> int:
    profile, lay = _profile_and_layout(args)
    runtime_path = _runtime_path(lay)
    runtime_config = _load_runtime(runtime_path)

    if not runtime_config.get("logged_in") and "tunnel" not in runtime_config:
        log.info("Already logged out.")
        return 0

    runtime_config.pop("tunnel", None)
    runtime_config["logged_in"] = False
    _save_runtime(runtime_path, runtime_config)
    log.info("Removed pairing from %s", runtime_path)

    _restart_service(profile.os_name, args.system)
    print()
    print("Logged out. llama-server and the tunnel have stopped.")
    return 0


def run_status(_args: argparse.Namespace) -> int:
    url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/status"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        log.error(
            "Could not reach the aipotluck service at %s (%s). Is it installed and running?",
            url, exc,
        )
        return 1

    logged_in = bool(payload.get("logged_in"))
    print(f"Login:        {'logged in' if logged_in else 'logged out'}")

    llama = payload.get("llama_server")
    if llama and not llama.get("error"):
        print(f"llama-server: {llama}")
    elif logged_in:
        print(f"llama-server: not running ({llama.get('error') if llama else 'unknown'})")
    else:
        print("llama-server: stopped (logged out)")

    tunnel = payload.get("tunnel")
    if tunnel:
        print(f"Tunnel:       {tunnel}")
        # A running newt process is not the same claim as a connected tunnel -- newt retries
        # forever on its own and never exits just because it can't reach Pangolin, so a real
        # failure here looks exactly like success unless this is checked explicitly (confirmed
        # live: 20+ hours of "running: true" with the tunnel never actually up).
        if tunnel.get("running") and tunnel.get("tunnel_connected") is False:
            print(
                "  WARNING: newt is running but has not established a tunnel connection -- "
                "check ~/.local/state/aipotluck/logs/newt.log for the reason (a wrong/unreachable "
                "endpoint is the common one)"
            )
    else:
        print("Tunnel:       " + ("not running (unexpected while logged in)" if logged_in else "stopped (logged out)"))

    # See CLAUDE.md's "Runtime parameters" convention: anything this project computes
    # automatically (not just what the user passed at install time) must be traceable here, not
    # just over HTTP -- this reads the exact same `runtime_params` GET /status already returns,
    # never a second, potentially-drifting summary of it. Router mode (CUR-1965's follow-up) means
    # there's no single active model to summarize -- print router-level params, then every SIZED
    # model with its own ctx_size/parallel/cache_type and why.
    params = payload.get("runtime_params")
    if params:
        router_shown = ", ".join(
            f"{field}={params[field]}"
            for field in ("gpu_layers", "models_max")
            if params.get(field) is not None
        )
        print(f"Router params: {router_shown or '(none set)'}")

        models = params.get("models") or {}
        if not models:
            print("Sized models: (none yet -- `pull` a model to size it)")
        else:
            # One line, same convention as the llama-server/Tunnel lines above -- the full
            # per-model reasoning (the "tuning" reasons) is deliberately left out here; read it
            # from GET /capabilities when you actually need it.
            shown = {model_id: {k: v for k, v in mp.items() if k != "tuning"} for model_id, mp in models.items()}
            print(f"Sized models: {shown}")

    return 0


def _reload_router_models(llama_cfg: dict) -> bool:
    """Best-effort: asks a currently-running router to re-scan the HF cache + --models-preset file
    (`GET /models?reload=1`, confirmed against server-models.cpp) so a model `pull` just
    downloaded/sized is visible immediately -- without this, a long-running router wouldn't notice
    a new cache entry until its next restart, since cache scanning happens at startup/reload, not
    per-request. Returns False (not an error -- the common case is simply "not logged in / service
    not running right now") whenever the router can't be reached; the next service start picks up
    everything from a full scan regardless, so nothing is lost by skipping this."""
    host = llama_cfg.get("host", "127.0.0.1")
    port = llama_cfg.get("port", 8080)
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/models?reload=1", timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _bench_binary(llama_cfg: dict) -> Path | None:
    """Locates llama-bench, which ships beside llama-server in every prebuilt llama.cpp release
    archive (fetch.py extracts the whole archive, it doesn't cherry-pick). Its absence is expected
    rather than broken: source_build.py builds only the llama-server target, so the arm64+CUDA and
    old-glibc devices that take that path won't have it, and the caller measures through the
    running server instead."""
    server_binary = llama_cfg.get("server_binary")
    if not server_binary:
        return None
    candidate = Path(server_binary).parent / ("llama-bench.exe" if os.name == "nt" else "llama-bench")
    return candidate if candidate.exists() else None


def _print_perf_verdict(result: model_perf.PerfResult) -> None:
    if result.grade == model_perf.GRADE_REFUSE:
        print(f"  {result.grade.upper()} -- {result.reason}")
        return
    print(
        f"  {result.grade.upper()} -- can produce ~{result.n_out:,} output tokens in budget "
        f"({result.decode_tokens_per_second:.1f} tok/s at a {result.ctx_size:,}-token context)"
    )
    print(f"  {result.reason}")


def _delete_cached_model(llama_cfg: dict, model_id: str) -> bool:
    """Removes a model's files through the running router's own `DELETE /models` (server.cpp:243 ->
    server-models.cpp's del_router_models -> common_download_remove), which stops any running
    instance and then clears the snapshot entry, its symlinks and the newly-orphaned blobs using
    llama.cpp's own cache logic rather than a reimplementation of its layout here.

    This is the only thing that actually enforces a refusal. The router auto-discovers everything
    in the HF cache and `--no-models-autoload` is a global switch rather than a per-model one, so
    merely withholding a preset would leave a refused model listed by /v1/models and selectable in
    the web app's picker -- exactly the outcome the grade exists to prevent."""
    host = llama_cfg.get("host", DEFAULT_HOST)
    port = llama_cfg.get("port", DEFAULT_PORT)
    url = f"http://{host}:{port}/models?model={urllib.parse.quote(model_id, safe='')}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="DELETE"), timeout=60) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError) as exc:
        log.warning("Could not remove %s through the running router: %s", model_id, exc)
        return False


def _benchmark_model(
    llama_cfg: dict, presets_path: Path, model_id: str, *, budget_seconds: float
) -> model_perf.PerfResult | None:
    """Measures and grades one already-downloaded model, persisting the verdict. Returns None when
    no trustworthy measurement was possible -- a missing grade, never a guessed one."""
    preset_args = model_presets.read_all(presets_path).get(model_id)
    if not preset_args or not preset_args.get("ctx-size"):
        log.warning(
            "%s has no sized preset, so there's no operating point to benchmark at -- skipping the "
            "speed check.", model_id,
        )
        return None

    host = llama_cfg.get("host", DEFAULT_HOST)
    port = llama_cfg.get("port", DEFAULT_PORT)
    base_url = f"http://{host}:{port}"

    # Free whatever the router is holding -- not just this model. Under --models-max 1 it keeps one
    # whole model resident, and llama-bench is about to load its own copy on top; on a 16GB Jetson
    # that was an outright allocation failure (NvMapMemAllocInternalTagged error 12) for models that
    # benchmark fine on an idle box. Unloading only the target is not enough, because the router is
    # frequently holding a different one. It reloads on the next request.
    model_perf_live.free_router_memory(base_url)

    if not model_perf.wait_until_idle():
        log.warning(
            "This machine is still busy, so a speed measurement would describe the contention "
            "rather than the model -- skipping. Re-run `aipotluck-local-client benchmark %s` when "
            "it's idle.", model_id,
        )
        return None

    ctx_size = int(preset_args["ctx-size"])
    bench = _bench_binary(llama_cfg)
    started = time.monotonic()
    try:
        if bench is not None:
            result = model_perf.probe_performance(
                bench, model_id,
                ctx_size=ctx_size,
                cache_type_k=preset_args.get("cache-type-k"),
                cache_type_v=preset_args.get("cache-type-v"),
                gpu_layers=llama_cfg.get("gpu_layers"),
                budget_seconds=budget_seconds,
            )
        else:
            # A from-source install (arm64+CUDA, old glibc) builds llama-server alone, so measure
            # through the running router instead. Same numbers, noisier path, larger safety margin
            # -- see model_perf_live. Skipping outright would leave the slowest hardware we support
            # as the only hardware with no speed check at all.
            log.info("No llama-bench in this install -- measuring through the running server instead.")
            result = model_perf_live.probe_performance_live(
                base_url, model_id, ctx_size=ctx_size, budget_seconds=budget_seconds,
            )
    except (model_perf.ModelPerfError, ValueError) as exc:
        log.warning("Could not measure %s on this device (%s) -- no grade recorded.", model_id, exc)
        return None

    model_perf_store.write_record(
        presets_path, model_id, result,
        llama_config=llama_cfg, preset_args=preset_args,
        probe_seconds=time.monotonic() - started,
    )
    return result


def run_pull_model(args: argparse.Namespace) -> int:
    """Downloads `args.model` via llama-server's own -hf downloader (see model_pull.py), sizes it
    (aipotluck.installer.model_sizing -- ctx_size/parallel/cache_type_k/-v, written into the
    router's --models-preset INI file), then asks a running router to pick both up immediately.
    There's no "active model" to set anymore -- llama-server's router mode (CUR-1965) serves
    whichever model a request's own "model" field names, autoloading on demand; `pull` just makes
    sure that works well (downloaded + correctly sized) the first time it's actually requested.
    Independent of login state: pulling a model doesn't need (or touch) pairing."""
    _profile, lay = _profile_and_layout(args)
    runtime_path = _runtime_path(lay)
    runtime_config = _load_runtime(runtime_path)

    llama_cfg = runtime_config.get("llama_cpp")
    if not llama_cfg or not llama_cfg.get("server_binary"):
        log.error("No llama.cpp install found at %s -- run the installer first.", runtime_path)
        return 1

    print(f"Pulling {args.model} -- this can take a while for a large quant.")
    try:
        pull_model(
            Path(llama_cfg["server_binary"]),
            args.model,
            gpu_layers=llama_cfg.get("gpu_layers"),
            timeout=args.timeout if args.timeout is not None else DEFAULT_TIMEOUT_SECONDS,
        )
    except ModelPullError as exc:
        log.error("%s", exc)
        return 1

    print("Sizing runtime parameters for this model...")
    presets_path = llama_cfg.get("presets_path")
    if not presets_path:
        log.warning("No presets_path configured (re-run the installer to pick this up) -- skipping sizing.")
    else:
        try:
            sizing = ensure_preset(
                Path(llama_cfg["server_binary"]), Path(presets_path), args.model,
                model_hf=args.model, gpu_layers=llama_cfg.get("gpu_layers"), force=True,
            )
        except ModelSizingError as exc:
            log.warning(
                "Automatic runtime sizing failed (%s) -- %s will run with llama-server's own "
                "defaults until this is retried.", exc, args.model,
            )
        else:
            # ensure_preset already persisted everything (the --models-preset INI section and its
            # sibling tuning-reasons JSON) -- nothing for this CLI to write back into runtime.json.
            # Writing a second copy here is exactly the "second summary that could drift" CLAUDE.md's
            # "Runtime parameters" convention warns against; diagnostics.runtime_params reads the
            # preset files directly (see model_presets.read_all/read_tuning).
            print(
                f"  ctx_size={sizing.ctx_size} parallel={sizing.parallel} "
                f"cache_type_k={sizing.cache_type_k} cache_type_v={sizing.cache_type_v}"
            )

    result = None
    if presets_path and not args.skip_benchmark:
        print(f"Measuring how fast this model runs here (up to {int(model_perf.PROBE_BUDGET_SECONDS)}s)...")
        result = _benchmark_model(
            llama_cfg, Path(presets_path), args.model,
            budget_seconds=model_perf.PROBE_BUDGET_SECONDS,
        )
        if result is not None:
            _print_perf_verdict(result)

    if result is not None and result.grade == model_perf.GRADE_REFUSE and not args.force:
        print()
        print(f"{args.model} is too slow to hold a conversation on this device.")
        removed = _delete_cached_model(llama_cfg, args.model)
        if removed:
            model_perf_store.forget(Path(presets_path), args.model)
            print("It has been removed so it can't be picked in the chat model list.")
        else:
            print(
                "It is still on disk -- the service wasn't reachable to remove it. Start the "
                f"service and run `aipotluck-local-client pull {args.model}` again, or keep it "
                "anyway with --force."
            )
        print("Pull it again with --force if you want it regardless.")
        return 1

    if _reload_router_models(llama_cfg):
        print()
        print(f"{args.model} is downloaded, sized, and live -- the router picked it up immediately.")
    else:
        print()
        print(
            f"{args.model} is downloaded and sized. The service isn't reachable right now (not "
            "logged in, or not running) -- it'll pick this model up the next time it starts."
        )
    return 0


def run_list_models(args: argparse.Namespace) -> int:
    """Lists every model already cached locally (see model_pull.list_cached_models). Doesn't need
    the device logged in or the service running -- this only reads the local HF-hub-compatible
    cache directory via llama-server's own --cache-list, independent of pairing/service state."""
    _profile, lay = _profile_and_layout(args)
    runtime_path = _runtime_path(lay)
    runtime_config = _load_runtime(runtime_path)

    llama_cfg = runtime_config.get("llama_cpp")
    if not llama_cfg or not llama_cfg.get("server_binary"):
        log.error("No llama.cpp install found at %s -- run the installer first.", runtime_path)
        return 1

    try:
        models = list_cached_models(Path(llama_cfg["server_binary"]))
    except ModelPullError as exc:
        log.error("%s", exc)
        return 1

    if not models:
        print("No models cached locally yet -- pull one with `aipotluck-local-client pull <repo[:quant]>`.")
        return 0

    # Router mode (CUR-1965's follow-up): there's no single "active" model anymore -- any cached
    # model can be requested at any time. "(sized)" instead marks whether `pull` (or the service's
    # own startup backfill) has already computed a ctx_size/parallel/cache_type preset for it.
    presets_path = llama_cfg.get("presets_path")
    sized = model_presets.known_model_ids(Path(presets_path)) if presets_path else set()
    graded = model_perf_store.read_all(Path(presets_path)).get("models", {}) if presets_path else {}

    print(f"{len(models)} model(s) cached locally:")
    for target in models:
        notes = []
        if target not in sized:
            notes.append("not sized yet")
        record = graded.get(target)
        if record is None:
            grade_label = "  ?     "
            notes.append("not benchmarked -- run `benchmark`")
        else:
            grade_label = f"  {record.get('grade', '?').upper():<6s}"
            n_out = record.get("n_out")
            if isinstance(n_out, int):
                notes.append(f"~{n_out:,} output tokens in budget")
            ctx = record.get("ctx_size")
            if isinstance(ctx, int):
                notes.append(f"{ctx:,}-token context")
            tps = record.get("decode_tokens_per_second")
            if isinstance(tps, (int, float)):
                notes.append(f"{tps:.1f} tok/s")
            if model_perf_store.is_stale(Path(presets_path), record, llama_config=llama_cfg):
                notes.append("stale -- re-run `benchmark`")
        suffix = f"  ({', '.join(notes)})" if notes else ""
        print(f"{grade_label}  {target}{suffix}")
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    """Re-measures one model, or every cached model, without re-downloading anything. This is the
    way back from a grade that was skipped (a busy machine) or has gone stale (a llama.cpp upgrade,
    a GPU-layers change, a re-size)."""
    _profile, lay = _profile_and_layout(args)
    runtime_config = _load_runtime(_runtime_path(lay))

    llama_cfg = runtime_config.get("llama_cpp")
    if not llama_cfg or not llama_cfg.get("server_binary"):
        log.error("No llama.cpp install found at %s -- run the installer first.", _runtime_path(lay))
        return 1
    presets_path = llama_cfg.get("presets_path")
    if not presets_path:
        log.error("No presets_path configured -- re-run the installer to pick this up.")
        return 1

    if args.model:
        targets = [args.model]
    else:
        try:
            targets = list_cached_models(Path(llama_cfg["server_binary"]))
        except ModelPullError as exc:
            log.error("%s", exc)
            return 1
        if not targets:
            print("No models cached locally yet -- pull one first.")
            return 0

    failures = 0
    for target in targets:
        print(f"{target}:")
        # Re-size before re-measuring. The grade is capped by the effective context -- the smaller
        # of what the model was trained for and what this device's memory can actually serve -- so
        # a stale ctx_size would silently cap the grade at whatever was true when the model was
        # first pulled. Anything that moves that number (a llama.cpp upgrade, a GPU-layers change,
        # RAM added or freed) should be picked up here rather than quietly ignored.
        try:
            sizing = ensure_preset(
                Path(llama_cfg["server_binary"]), Path(presets_path), target,
                model_hf=target, gpu_layers=llama_cfg.get("gpu_layers"), force=True,
            )
        except ModelSizingError as exc:
            log.warning(
                "Could not re-size %s (%s) -- grading against whatever preset it already had.",
                target, exc,
            )
        else:
            if sizing is not None:
                print(f"  re-sized: ctx_size={sizing.ctx_size} cache_type_k={sizing.cache_type_k} "
                      f"cache_type_v={sizing.cache_type_v}")

        result = _benchmark_model(
            llama_cfg, Path(presets_path), target, budget_seconds=model_perf.PROBE_BUDGET_SECONDS,
        )
        if result is None:
            failures += 1
        else:
            _print_perf_verdict(result)
    return 1 if failures and len(targets) == 1 else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        if args.command == "login":
            return run_login(args)
        if args.command == "logout":
            return run_logout(args)
        if args.command == "status":
            return run_status(args)
        if args.command == "pull":
            return run_pull_model(args)
        if args.command == "list":
            return run_list_models(args)
        if args.command == "benchmark":
            return run_benchmark(args)
    except SystemExit:
        raise
    except Exception as exc:  # top-level guard: always report clearly
        log.error("%s failed: %s", args.command, exc, exc_info=args.verbose)
        return 1
    return 1  # unreachable: argparse's `required=True` on the subparser rules this out


if __name__ == "__main__":
    raise SystemExit(main())
