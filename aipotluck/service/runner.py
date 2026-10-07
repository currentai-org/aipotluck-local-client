"""Portable service runtime shared by every OS entry point.

This module contains zero OS-specific code (no signal handling, no service
framework imports) so the exact same logic runs under:

- Linux:   systemd unit invokes aipotluck/service/aipotluck_service.py directly
- macOS:   launchd invokes aipotluck/service/aipotluck_service.py directly
- Windows: schtasks (no-admin) invokes aipotluck/service/aipotluck_service.py directly;
           the --system pywin32 Windows Service
           (aipotluck/service/windows_service_host.py) imports AipotluckServiceRunner
           and drives start()/stop() from SvcDoRun/SvcStop instead of a
           signal handler.

AipotluckServiceRunner owns:
  - the HTTP status server (GET /healthz, GET /status)
  - the LlamaSupervisor lifecycle (start on run, stop on shutdown)
  - the NewtSupervisor lifecycle (same as above)

Both supervisors are gated on runtime.json's "logged_in" flag: every fresh install starts logged
out (no Pangolin credentials), and neither llama-server nor the tunnel is started until
`aipotluck-local-client login` sets logged_in true and restarts the service. `logout` reverses it.
See aipotluck/installer/cli.py.

It exposes start() / stop() / wait() so callers control the lifecycle
without needing to know how each OS delivers a "please stop" signal.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aipotluck import diagnostics  # noqa: E402
from aipotluck.diagnostics import runtime_params  # noqa: E402, F401 -- re-exported, see below
from aipotluck.installer import model_perf_live, model_pull, model_presets, model_sizing  # noqa: E402
from aipotluck.service.llama_supervisor import LlamaSupervisor  # noqa: E402
from aipotluck.service import jobs as jobs_module  # noqa: E402
from aipotluck.service import model_health, model_ops  # noqa: E402
from aipotluck.service.model_health import ModelHealthWatcher  # noqa: E402
from aipotluck.service.newt_supervisor import NewtSupervisor  # noqa: E402

SERVICE_NAME = "aipotluck"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8069  # distinct from llama-server's own default port 8080
DEFAULT_MODELS_MAX = 1  # see runner.py's build_llama_server_args and CLAUDE.md; one model loaded
                         # at a time keeps aipotluck.installer.model_sizing's "all budget, one
                         # model" memory math valid -- see that module's own docstring for why
                         # multi-model concurrency is deliberately not attempted yet.

log = logging.getLogger("aipotluck.service")


def load_runtime_config(config_dir: Path) -> dict:
    runtime_path = config_dir / "runtime.json"
    if not runtime_path.exists():
        log.warning("No runtime.json found at %s -- run the installer first", runtime_path)
        return {}
    try:
        return json.loads(runtime_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Failed to read runtime.json: %s", exc)
        return {}


def build_llama_server_args(llama_cfg: dict) -> list[str]:
    """Router mode (CUR-1965's follow-up): deliberately no `-hf`/`--model`/`--ctx-size`/
    `--parallel`/`--cache-type-k`/`-v` of our own. Passing no model at all puts llama-server into
    its own native multi-model router (confirmed against
    vendor/llama.cpp/tools/server/server-models.cpp): it auto-discovers every model already in the
    HF cache and routes each request by the `"model"` field in its body, loading/unloading
    instances on demand -- this is what makes "the user switched models" something llama-server
    itself detects and handles, not something this project has to intercept or proxy. Per-model
    ctx-size/parallel/cache-type-k/-v instead live in the `--models-preset` INI file
    (aipotluck.installer.model_presets/model_sizing write it), since the router applies those
    per model, not once globally."""
    args = ["--host", str(llama_cfg.get("host", "127.0.0.1")), "--port", str(llama_cfg.get("port", 8080))]

    presets_path = llama_cfg.get("presets_path")
    if presets_path:
        args += ["--models-preset", str(presets_path)]
    else:
        log.warning("No presets_path configured; llama-server's router will use its own defaults for every model")

    # Only set for a --model-path install: a local GGUF outside the HF cache is invisible to the
    # router's own cache auto-discovery, so it needs this second, explicit source (confirmed
    # against common_preset_context::load_from_models_dir -- a loose *.gguf directly in this
    # directory is routed under its filename minus ".gguf", which is exactly the id
    # model_sizing.ensure_preset is given for it at install time).
    models_dir = llama_cfg.get("models_dir")
    if models_dir:
        args += ["--models-dir", str(models_dir)]

    args += ["--models-max", str(llama_cfg.get("models_max", DEFAULT_MODELS_MAX))]

    # Global -- overlaid onto every model instance's own args by the router (confirmed against
    # server-models.cpp), so this is the one llama-server flag that still belongs at the router
    # level rather than per-model: it's a hardware capability (how many layers this box's GPU
    # backend can hold), not something that varies by which model is currently loaded.
    gpu_layers = llama_cfg.get("gpu_layers")
    if gpu_layers is not None:
        args += ["--gpu-layers", str(gpu_layers)]

    return args


def ensure_presets_file(llama_cfg: dict) -> None:
    """llama-server refuses to start at all when its `--models-preset` file doesn't exist
    ("preset file does not exist"), and nothing guarantees one does: the installer only creates it
    as a side effect of sizing the default model, so an install that skipped or failed that step
    left the router crash-looping. An empty file is a valid preset file with no sections."""
    presets_path = llama_cfg.get("presets_path")
    if not presets_path:
        return
    try:
        Path(presets_path).parent.mkdir(parents=True, exist_ok=True)
        Path(presets_path).touch(exist_ok=True)
    except OSError as exc:
        log.error("Could not create the presets file %s: %s", presets_path, exc)


def build_supervisor(runtime_config: dict, log_dir: Path | None) -> LlamaSupervisor | None:
    llama_cfg = runtime_config.get("llama_cpp")
    if not llama_cfg:
        log.error("runtime.json has no llama_cpp section; cannot supervise llama-server")
        return None

    server_binary = Path(llama_cfg["server_binary"])
    if not server_binary.exists():
        log.error("llama-server binary not found at %s", server_binary)
        return None

    return LlamaSupervisor(
        server_binary=server_binary,
        args=build_llama_server_args(llama_cfg),
        host=llama_cfg.get("host", "127.0.0.1"),
        port=llama_cfg.get("port", 8080),
        log_dir=log_dir,
    )


def models_missing_presets(llama_cfg: dict) -> list[str]:
    """Every model in the HF cache with no `--models-preset` section yet. Cheap -- one
    `--cache-list` call plus one INI read -- and empty in the common case, since `pull` always
    writes one. Never raises: a listing failure means nothing to backfill, not a broken service."""
    server_binary_str = llama_cfg.get("server_binary")
    presets_path_str = llama_cfg.get("presets_path")
    if not server_binary_str or not presets_path_str:
        return []
    try:
        cached = model_pull.list_cached_models(Path(server_binary_str))
    except model_pull.ModelPullError as exc:
        log.warning("Could not list cached models for preset backfill: %s", exc)
        return []
    known = model_presets.known_model_ids(Path(presets_path_str))
    return [model_id for model_id in cached if model_id not in known]


def backfill_missing_presets(llama_cfg: dict, *, progress=lambda _message: None) -> dict:
    """Sizes every cached model that has no `--models-preset` section (CUR-1965's "on pull OR on
    startup if it's found to be missing") -- recovering from a deleted/hand-edited-away presets
    file, or a model that reached the cache some other way -- then reloads the router so the new
    presets take effect.

    This runs as a background job after the service is up, never on the startup path: each sizing
    probe loads the model, which takes seconds to minutes, and `status` and the model list have to
    answer from the moment the service starts. The price is a short window in which the router
    serves an unsized model on llama-server's own defaults -- the same thing it does for any model
    that has no preset. As in `pull`, the router is emptied before each probe, so the probe is not
    competing with a resident model for the same memory.

    Never raises -- a sizing failure here leaves that model on llama-server's defaults, and is
    reported in the returned summary rather than deleting anything."""
    missing = models_missing_presets(llama_cfg)
    summary: dict = {"sized": [], "failed": {}, "router_reloaded": False}
    if not missing:
        return summary

    server_binary = Path(llama_cfg["server_binary"])
    presets_path = Path(llama_cfg["presets_path"])
    log.info("Backfilling missing runtime-sizing presets for %d cached model(s): %s", len(missing), missing)
    for model_id in missing:
        progress(f"sizing {model_id}")
        model_perf_live.free_router_memory(model_ops.base_url(llama_cfg))
        try:
            model_sizing.ensure_preset(
                server_binary, presets_path, model_id, model_hf=model_id, gpu_layers=llama_cfg.get("gpu_layers")
            )
        except model_sizing.ModelSizingError as exc:
            log.warning("Could not size cached model %r (%s) -- it will use llama-server's own defaults", model_id, exc)
            summary["failed"][model_id] = str(exc)
        else:
            summary["sized"].append(model_id)
    if summary["sized"]:
        summary["router_reloaded"] = model_ops.reload_router(llama_cfg)
    return summary


def build_health_watcher(
    runtime_config: dict, config_dir: Path, log_dir: Path | None
) -> ModelHealthWatcher | None:
    """`None` when there is no router to watch or no presets file to adjust -- without a
    --models-preset path there is no per-model context to shrink, so an OOM has no remedy this
    layer could apply."""
    llama_cfg = runtime_config.get("llama_cpp") or {}
    presets_path = llama_cfg.get("presets_path")
    if not llama_cfg.get("server_binary") or not presets_path:
        return None
    host = llama_cfg.get("host", DEFAULT_HOST)
    port = llama_cfg.get("port", DEFAULT_PORT)
    return ModelHealthWatcher(
        base_url=f"http://{host}:{port}",
        presets_path=Path(presets_path),
        config_dir=Path(config_dir),
        log_path=(Path(log_dir) / "llama-server.log") if log_dir else None,
    )


def build_newt_supervisor(runtime_config: dict, log_dir: Path | None) -> NewtSupervisor | None:
    """`None` when this install has no `tunnel` section -- true for every device until it's paired
    via `aipotluck-local-client login`, and again after `logout` drops the section."""
    tunnel_cfg = runtime_config.get("tunnel")
    if not tunnel_cfg:
        return None

    newt_binary = Path(tunnel_cfg["binary"])
    if not newt_binary.exists():
        log.error("newt binary not found at %s", newt_binary)
        return None

    return NewtSupervisor(
        newt_binary=newt_binary,
        tunnel_id=tunnel_cfg["id"],
        tunnel_secret=tunnel_cfg["secret"],
        tunnel_endpoint=tunnel_cfg["endpoint"],
        log_dir=log_dir,
    )


class _StatusHandler(BaseHTTPRequestHandler):
    """Bound to a running AipotluckServiceRunner via class attributes set
    at server construction time (http.server handlers are instantiated
    per-request, so state has to live on the class or be injected via a
    handler factory -- we use a factory, see AipotluckServiceRunner._make_handler)."""

    runner: "AipotluckServiceRunner"

    def log_message(self, format: str, *args) -> None:  # quiet default stderr logging
        log.info("%s - %s", self.address_string(), format % args)

    def _write_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _redacted_runtime_config(runtime_config: dict) -> dict:
        """/status is unauthenticated on localhost -- never echo the tunnel secret back out."""
        if "tunnel" not in runtime_config:
            return runtime_config
        sanitized = dict(runtime_config)
        sanitized["tunnel"] = {**runtime_config["tunnel"], "secret": "<redacted>"}
        return sanitized

    def _write_error(self, message: str, *, code: str, status: int) -> None:
        self._write_json({"error": {"code": code, "message": message}}, status=status)

    def _read_json_body(self) -> dict:
        """The request body as a dict. An empty body is an empty dict -- POST /models/benchmark
        with no arguments is a legitimate "measure everything" request."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        return payload

    def _llama_cfg(self) -> dict:
        return self.runner.runtime_config.get("llama_cpp") or {}

    def _query(self) -> dict[str, list[str]]:
        return urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

    def _route(self) -> str:
        return urllib.parse.urlparse(self.path).path.rstrip("/") or "/"

    def do_GET(self) -> None:
        if self._route() == "/models":
            try:
                self._write_json(
                    model_ops.list_models(self._llama_cfg(), self.runner.config_dir)
                )
            except model_ops.ModelOpError as exc:
                self._write_error(str(exc), code=exc.code, status=exc.status)
            return

        if self._route() == "/jobs":
            self._write_json({"jobs": [job.to_dict() for job in self.runner.jobs.list()]})
            return

        if self._route().startswith("/jobs/"):
            job = self.runner.jobs.get(self._route()[len("/jobs/"):])
            if job is None:
                self._write_error("No such job.", code="unknown_job", status=404)
            else:
                self._write_json(job.to_dict())
            return

        if self.path == "/healthz":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/status":
            logged_in = bool(self.runner.runtime_config.get("logged_in"))
            supervisor = self.runner.supervisor
            if supervisor:
                llama_info = supervisor.info().__dict__
            elif logged_in:
                llama_info = {"error": "supervisor not started"}
            else:
                llama_info = None
            newt_supervisor = self.runner.newt_supervisor
            tunnel_info = newt_supervisor.info().__dict__ if newt_supervisor else None
            self._write_json(
                {
                    "service": SERVICE_NAME,
                    "status": "running",
                    "logged_in": logged_in,
                    "runtime_config": self._redacted_runtime_config(self.runner.runtime_config),
                    "runtime_params": runtime_params(self.runner.runtime_config),
                    "llama_server": llama_info,
                    "tunnel": tunnel_info,
                    # Held-back models are reported here rather than only in the CLI, because the
                    # web app is where the user picks a model and is the only surface that can
                    # explain why one keeps failing at the moment they try it.
                    "held_back_models": model_health.quarantined_models(self.runner.config_dir),
                }
            )
            return

        if self.path == "/capabilities":
            # No secrets in here (unlike runtime_config's tunnel section, see _redacted_runtime_config
            # above) -- everything reported is host hardware/software facts and the installer's own
            # public decision logic, so this is unauthenticated on localhost the same as /status.
            fingerprint = diagnostics.gather_fingerprint(
                config_dir=self.runner.config_dir, runtime_config=self.runner.runtime_config
            )
            self._write_json(fingerprint)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        route = self._route()
        if route not in ("/models", "/models/benchmark"):
            self.send_response(404)
            self.end_headers()
            return

        try:
            body = self._read_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._write_error(f"Could not read the request body: {exc}", code="bad_request", status=400)
            return

        llama_cfg = self._llama_cfg()
        config_dir = self.runner.config_dir
        try:
            model_ops.require_llama(llama_cfg)
        except model_ops.ModelOpError as exc:
            self._write_error(str(exc), code=exc.code, status=exc.status)
            return

        if route == "/models":
            model_id = body.get("model")
            if not isinstance(model_id, str) or not model_id.strip():
                self._write_error(
                    'A "model" field is required, e.g. {"model": "org/repo:Q4_K_M"}.',
                    code="bad_request", status=400,
                )
                return
            # The preflight runs HERE, before the job is queued, so an oversized model is refused
            # in the response the caller is already waiting on rather than as a job that fails
            # minutes later. It is also the one rejection a caller can act on by retrying.
            force = bool(body.get("force"))
            allow_oversized = bool(body.get("allow_oversized")) or force
            if not allow_oversized:
                verdict = model_ops.check_size_before_download(model_id)
                if verdict is not None and not verdict["fits"]:
                    self._write_json(
                        {
                            "error": {
                                "code": "too_large_for_device",
                                "message": verdict["detail"],
                                "preflight": verdict,
                                "retry_with": {"allow_oversized": True},
                            }
                        },
                        status=409,
                    )
                    return

            keep_rejected = bool(body.get("keep_rejected")) or force
            timeout = body.get("timeout")

            def _pull(job):
                return model_ops.pull(
                    llama_cfg, config_dir, model_id,
                    allow_oversized=True,  # already decided above, with the caller's answer
                    keep_rejected=keep_rejected,
                    timeout=float(timeout) if timeout is not None else None,
                    progress=self.runner.jobs.progress_callback(job),
                    download_progress=self.runner.jobs.download_callback(job),
                )

            job = self.runner.jobs.submit("pull", model_id, _pull)
            self._write_json({"job": job.to_dict()}, status=202)
            return

        model_id = body.get("model")
        if model_id is not None and not isinstance(model_id, str):
            self._write_error('"model" must be a string if given.', code="bad_request", status=400)
            return

        def _benchmark(job):
            return model_ops.benchmark(
                llama_cfg, config_dir, model_id,
                progress=self.runner.jobs.progress_callback(job),
            )

        job = self.runner.jobs.submit("benchmark", model_id, _benchmark)
        self._write_json({"job": job.to_dict()}, status=202)

    def do_DELETE(self) -> None:
        if self._route() != "/models":
            self.send_response(404)
            self.end_headers()
            return
        # `?model=` rather than a path segment: a model id carries both "/" and ":", and this is
        # the exact shape llama.cpp's own router uses for the same operation (server-models.cpp's
        # DELETE /models), which this ultimately calls through to.
        model_id = (self._query().get("model") or [""])[0]
        if not model_id:
            self._write_error(
                "A ?model= query parameter is required.", code="bad_request", status=400,
            )
            return
        try:
            self._write_json(
                model_ops.remove(self._llama_cfg(), self.runner.config_dir, model_id)
            )
        except model_ops.ModelOpError as exc:
            self._write_error(str(exc), code=exc.code, status=exc.status)


class AipotluckServiceRunner:
    """OS-agnostic service body: HTTP status server + LlamaSupervisor.

    Usage:
        runner = AipotluckServiceRunner(host, port, config_dir, log_dir)
        runner.start()   # non-blocking: spawns the HTTP server + supervisor
        runner.wait()    # blocking: returns once stop() is called elsewhere
        runner.stop()    # idempotent; call from a signal handler or an
                          # SCM stop callback (pywin32 SvcStop)
    """

    def __init__(self, host: str, port: int, config_dir: Path, log_dir: Path | None) -> None:
        self.host = host
        self.port = port
        self.config_dir = config_dir
        self.log_dir = log_dir

        self.runtime_config: dict = {}
        self.supervisor: LlamaSupervisor | None = None
        self.newt_supervisor: NewtSupervisor | None = None
        self.health_watcher: ModelHealthWatcher | None = None
        self.jobs = jobs_module.JobRunner()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._stopped = threading.Event()

    def start(self) -> None:
        """Returns as soon as everything is started, not finished: the HTTP server is bound first,
        so `status` answers from the very start, and nothing that can take seconds (spawning the
        router, connecting the tunnel, sizing models) runs on this thread."""
        self.runtime_config = load_runtime_config(self.config_dir)

        runner = self

        class BoundHandler(_StatusHandler):
            pass

        BoundHandler.runner = runner

        self._server = ThreadingHTTPServer((self.host, self.port), BoundHandler)
        log.info("aipotluck service listening on http://%s:%s", self.host, self.port)

        # Started regardless of login state: model management is independent of pairing, the
        # same way `pull` and `list` are on the CLI.
        self.jobs.start()

        self._server_thread = threading.Thread(
            target=self._server.serve_forever, name="aipotluck-http", daemon=True
        )
        self._server_thread.start()

        # Every fresh install is logged out (aipotluck.installer.install writes "logged_in": false and no
        # "tunnel" section) -- hold both llama-server and newt back entirely until `python -m
        # `aipotluck-local-client login` flips this and restarts the service. This is what lets the public
        # one-line installer take zero arguments: it never needs credentials in hand to finish.
        if self.runtime_config.get("logged_in"):
            llama_cfg = self.runtime_config.get("llama_cpp") or {}
            ensure_presets_file(llama_cfg)

            self.supervisor = build_supervisor(self.runtime_config, self.log_dir)
            if self.supervisor:
                log.info("Starting llama-server supervisor")
                self.supervisor.start()
            else:
                log.error("llama-server supervisor could not be created; service will run without it")

            self.newt_supervisor = build_newt_supervisor(self.runtime_config, self.log_dir)
            if self.newt_supervisor:
                log.info("Starting newt supervisor")
                self.newt_supervisor.start()

            # Started last and stopped first: it only observes, so it is never the reason anything
            # else is unavailable.
            self.health_watcher = build_health_watcher(
                self.runtime_config, self.config_dir, self.log_dir
            )
            if self.health_watcher:
                log.info("Starting model health watcher")
                self.health_watcher.start()

            # Queued behind nothing at startup, and ahead of any pull or benchmark that arrives
            # while it runs -- the same one-at-a-time queue, so a probe never shares memory with
            # one of theirs. It shows up in GET /jobs like any other job.
            if models_missing_presets(llama_cfg):
                self.jobs.submit(
                    "backfill", None,
                    lambda job: backfill_missing_presets(
                        llama_cfg, progress=self.jobs.progress_callback(job)
                    ),
                )
        else:
            log.info(
                "Device is logged out -- llama-server and the tunnel will not start until you run "
                "`aipotluck-local-client login`."
            )

    def wait(self) -> None:
        """Block the calling thread until stop() has fully completed."""
        self._stopped.wait()

    def stop(self, timeout: float = 20.0) -> None:
        if self._stopped.is_set():
            return
        log.info("Stopping aipotluck service")
        if self._server:
            self._server.shutdown()
            if self._server_thread:
                self._server_thread.join(timeout=5)
            self._server.server_close()
            self._server = None
        self.jobs.stop()
        if self.health_watcher:
            self.health_watcher.stop()
        if self.supervisor:
            self.supervisor.stop(timeout=timeout)
        if self.newt_supervisor:
            self.newt_supervisor.stop(timeout=timeout)
        log.info("aipotluck service stopped")
        self._stopped.set()
