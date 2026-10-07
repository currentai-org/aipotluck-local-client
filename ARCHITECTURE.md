# aipotluck-local-client — Architecture Plan

Status: PLANNING (not yet implemented). This doc captures findings from
examining llama.cpp's install methods and proposes how we wrap it.

## 1. What llama.cpp actually gives us to wrap

llama.cpp has **no unified installer script**. `docs/install.md` lists four
paths, none of which is "run this script":

| Method      | Win | Mac | Linux | Notes |
|-------------|-----|-----|-------|-------|
| conda-forge | Y   | Y   | Y     | needs conda/mamba/pixi already present |
| winget      | Y   |     |       | Windows only |
| Homebrew    |     | Y   | Y (linuxbrew) | needs brew already present |
| Nix         |     | Y   | Y     | niche |
| build from source | Y | Y | Y | `cmake -B build && cmake --build build --config Release` |

All of the package-manager paths ultimately resolve to the same artifact:
GitHub Releases at `ggml-org/llama.cpp`, tagged `bNNNNN` (build number, not
semver). Each release publishes a flat archive per platform/backend, e.g.:

```
llama-b10989-bin-macos-arm64.tar.gz
llama-b10989-bin-macos-x64.tar.gz
llama-b10989-bin-ubuntu-x64.tar.gz
llama-b10989-bin-ubuntu-arm64.tar.gz
llama-b10989-bin-ubuntu-cuda-12.8-x64.tar.gz
llama-b10989-bin-ubuntu-vulkan-x64.tar.gz
llama-b10989-bin-ubuntu-rocm-10.0-x64.tar.gz
llama-b10989-bin-win-cpu-x64.zip
llama-b10989-bin-win-cuda-13.4-x64.zip
llama-b10989-bin-win-vulkan-x64.zip
llama-b10989-bin-win-cpu-arm64.zip
... (sycl, openvino, s390x, android, xcframework, cudart runtime bundles, ui)
```

Confirmed by extracting `llama-b10989-bin-ubuntu-x64.tar.gz`: it's a flat
directory — `llama-server`, `llama-cli`, `llama-quantize`, `llama-bench`, a
pile of `libggml-cpu-<arch feature level>.so` variants (dynamic CPU dispatch),
`libllama*.so`, `libmtmd.so`, `LICENSE`. No install script, no manifest. You
place it somewhere and run the binary directly with `LD_LIBRARY_PATH`
pointing at the same dir (or keep binaries+libs colocated, which works
out-of-the-box on Linux/macOS via rpath, and via same-dir DLL search on
Windows).

**Conclusion:** "the llama.cpp installer" we stick close to is really just:
*download the right release asset for this OS/arch/backend, verify it,
extract it, keep it out of the way of user data*. That's exactly what brew's
formula and the winget manifest do under the hood. We replicate that logic
ourselves rather than shelling out to brew/winget, because:

- brew/winget/conda are not guaranteed present (chicken-and-egg for a
  cross-platform *installer*), and their presence/version drift is outside
  our control.
- Building from source requires a C++ toolchain we can't assume end users
  have — bad for "reduce friction."
- The GitHub Release binaries are the one path guaranteed on all three OSes
  with zero prerequisites beyond network access.

We keep an **escape hatch**: if the user already has `brew`/`winget`
available and prefers system-managed installs, offer `--via brew` /
`--via winget` flags that just shell out. Default path is our own
download+verify+extract, because it's the one we can make deterministic,
pinned, and offline-repeatable (cache the archive).

## 2. Version pinning strategy

- Pin a specific release tag (e.g. `b10989`) in a config file
  (`llama_version.json` at repo root), not "latest" — reproducibility.
- `scripts/update_llama_version.py` (later) bumps the pin deliberately.
- Submodule `vendor/llama.cpp` tracks source at that same tag for reference /
  future custom builds, but the installer does **not** build from the
  submodule by default — it downloads the matching prebuilt release. The
  submodule exists so we can (a) read server CLI flags/docs programmatically,
  (b) fall back to source build when no prebuilt asset matches the host
  (e.g. exotic Linux arch), (c) later carry patches if we ever need a custom
  build.

## 3. Installer wrapper design

```
aipotluck-local-client/
  vendor/llama.cpp/            # git submodule, pinned commit
  llama_version.json           # {"tag": "b10989", "assets": {...sha256s...}}
  aipotluck/                   # the root package -- see README.md's "Repo layout" for the full,
    installer/                 # current-state tree (this diagram predates the `aipotluck.` rename
      __init__.py               # and login/logout/status CLI; kept here for the original design
      platform_detect.py        # rationale, not as an up-to-date file listing).
      fetch.py                  # download+checksum+extract release asset
      layout.py                  # where things live on disk per-OS
      install.py                 # CLI entry: `python -m aipotluck.installer.install`
      service/
        __init__.py
        base.py                  # ServiceManager ABC: install/uninstall/start/stop/status
        systemd.py                # Linux
        launchd.py                # macOS
        windows_service.py        # Windows
    service/
      aipotluck_service.py       # the "blank" python service payload (for now: no-op / health loop)
  packaging/
    windows/                   # future: Inno Setup / MSI wrapper
    macos/                     # future: .pkg / notarization
    linux/                     # future: .deb/.rpm/AppImage
  ARCHITECTURE.md
```

### 3.1 Platform/arch/backend detection (`platform_detect.py`)
- `platform.system()` → `Linux` / `Darwin` / `Windows`
- `platform.machine()` → normalize to `x64` / `arm64`
- GPU backend probe (best-effort, non-fatal if inconclusive):
  - NVIDIA: `nvidia-smi` present → cuda asset
  - AMD: `rocminfo` / `/dev/kfd` → rocm asset (Linux only for now)
  - Apple Silicon: Metal is built into the `macos-arm64` asset already (no
    separate variant) — nothing to detect, just pick macos-arm64.
  - Vulkan: generic fallback for non-NVIDIA/AMD GPU Linux/Windows if desired.
  - Default/no GPU detected → plain CPU asset (`ubuntu-x64`, `win-cpu-x64`,
    `macos-*`). CPU asset ships multiple `libggml-cpu-<isa>.so` variants and
    llama.cpp dynamically dispatches to the best one at runtime — so plain
    CPU asset is always a safe baseline even when GPU detection is wrong.
- Asset-name mapping table lives in `llama_version.json` so it's data, not
  code, and survives across version bumps without touching logic.

### 3.2 Fetch + verify (`fetch.py`)
- Download from `https://github.com/ggml-org/llama.cpp/releases/download/<tag>/<asset>`.
- Verify sha256 against `llama_version.json` (we compute and pin these
  ourselves at version-bump time — llama.cpp releases don't publish
  checksums, so we generate our own manifest and treat first-download-per-tag
  as trust-on-first-use, recorded for future audits).
- Extract to a versioned dir so multiple versions can coexist:
  `<install_root>/llama.cpp/<tag>/<platform-asset>/`.
- Idempotent: re-running with the same tag/asset is a no-op if already
  extracted and checksum matches.

### 3.2.1 When no prebuilt asset works: source build + our own binary cache

Not every host has a viable upstream release asset. `build_strategy.py` decides this per host --
two independent, confirmed-real triggers: no asset was ever published for the detected
os/arch/backend (every Jetson-class arm64+CUDA board -- `nvidia-smi` is real there, but llama.cpp's
release CI has never published a `linux-arm64-cuda` asset), or the best available asset needs a
newer glibc than the host has (llama.cpp's own CI builds its Linux arm64 assets against a newer
base image than most arm64 boards run -- the pinned tag's `linux-arm64-cpu` asset needs glibc 2.38,
JetPack 6.2.3 ships 2.35, so even the CPU-only fallback refuses to start there).

When that happens, three tiers are tried in order, each cheaper than the next:

1. **Our own binary cache** (`build_cache.py`, `llama_custom_builds.json`). Once a binary has been
   built from source for a given (os, arch, backend, GPU architecture) combination, there's no
   reason to pay a 30-90 minute compile again on every matching host -- it's archived in the same
   format as the upstream release assets (a `.tar.gz` extracting to a flat top-level directory,
   matching llama.cpp's own release packaging exactly) and hosted as a GitHub release on this
   project's own public repo. The manifest shape deliberately mirrors `llama_version.json`'s assets
   section (`release_base_url` + `tag` + `{key: {file, sha256}}`), so this reuses
   `fetch.download_asset`/`extract_archive`/`find_binary` completely unchanged -- only the key
   resolution differs. The key includes the CUDA architecture where upstream's never need to
   (`linux-arm64-cuda-sm87` for Orin, distinct from a hypothetical `-sm72` for Xavier or `-sm53` for
   the original Nano/TX1) -- a binary built for one `CMAKE_CUDA_ARCHITECTURES` value will not run
   correctly on a different one, so these are never treated as interchangeable.
2. **Build from source** (`source_build.py`), when neither an upstream asset nor a cached custom
   one matches. Checks the host actually has what a build needs (cmake, git, a C++ compiler,
   OpenSSL headers, the OpenMP runtime, and per-backend a GPU vendor toolchain -- nvcc for CUDA,
   glslc+headers for Vulkan, hipcc for ROCm), cross-referenced against every Linux job in
   llama.cpp's own release CI rather than guessed from one host's missing package. For the
   apt-fixable gaps, the installer can install them itself via `sudo apt-get install`, but only
   with explicit consent (`--allow-apt-install`, or an interactive y/N prompt that never fires
   without a real controlling terminal) -- a GPU vendor toolchain is never auto-installed under any
   flag, since none of those are a small, safe, distro-generic install. Two build-time gotchas are
   the same shape (a plain, non-`REQUIRED` `find_package(...)` that degrades silently rather than
   failing the configure step): missing OpenSSL headers silently compile out HTTPS support (which
   `-hf`/`--cache-list`, and therefore this project's own pull/list commands, depend on), and
   missing libgomp1 silently falls back to single-threaded CPU inference. Both are caught in
   preflight, before ever starting a build that can otherwise take well over an hour on a low-power
   board. The build sets `CMAKE_INSTALL_RPATH=$ORIGIN` (matching llama.cpp's own release CI)
   specifically so its output is relocatable -- movable into the binary cache above, or into a
   different install root -- rather than only working from the exact path it was compiled at.
3. Refuse clearly, if source-build prerequisites are missing and the caller declined
   (or was never asked, e.g. `--no-apt-install`/no terminal) to fix them, or `--no-source-build` was
   passed to fail fast rather than sit through a long build.

Confirmed end-to-end on a real Jetson Orin Nano (JetPack 6.2.3): the full pipeline (detection ->
strategy -> preflight -> source build) produced a genuinely working CUDA `llama-server`
(`--list-devices` reports `CUDA0: Orin`), and once cached, the same host skips the ~35 minute
compile entirely on a repeat install.

### 3.3 Install locations (`layout.py`), per OS, no-admin-by-default
- Linux: `~/.local/share/aipotluck/` (binaries+libs), config in
  `~/.config/aipotluck/`, logs in `~/.local/state/aipotluck/logs/`
  (XDG dirs).
- macOS: `~/Library/Application Support/aipotluck/`, logs in
  `~/Library/Logs/aipotluck/`.
- Windows: `%LOCALAPPDATA%\aipotluck\` for binaries/config,
  `%LOCALAPPDATA%\aipotluck\logs\`.
- A `--system` flag can later target machine-wide install paths
  (`/opt/aipotluck`, `/Library/Application Support`,
  `%ProgramData%\aipotluck`) for the real "system service" case, requiring
  elevation. Default stays per-user/no-admin to minimize friction, matching
  llama.cpp's own brew/winget UX (no admin prompt).

### 3.4 Top-level installer flow (`install.py`)
1. Detect platform/arch/backend.
2. Resolve asset name from `llama_version.json`.
3. Fetch+verify+extract llama.cpp release into versioned install dir.
4. Write/refresh a small `runtime.json` pointing at the resolved
   `llama-server` binary path + lib dir for this OS.
5. Install our Python service (section 4) pointed at that `runtime.json`.
6. Optionally start the service immediately (`--start` flag, default on).
7. Print a summary: install dir, service status, server URL
   (`http://127.0.0.1:8080` by default, matching llama.cpp's own default so
   we don't invent new conventions).

CLI surface (argparse, no exotic deps):
```
python -m aipotluck.installer.install [--tag b10989] [--backend auto|cpu|cuda|vulkan|rocm]
                             [--install-dir PATH] [--system] [--no-start]
                             [--via auto|direct|brew|winget]
```

## 4. Python service injection design

"Tack on our bare python service" after the llama.cpp install finishes.
Scope for now: **blank/no-op service** — just proves out the cross-platform
service lifecycle (install, autostart, start/stop/status, logs) so the real
transport layer has a place to live later. It should NOT try to manage
llama-server itself yet beyond optionally shelling out to start it — keep
concerns separated: llama.cpp does inference serving, our service is future
transport/orchestration surface.

### 4.1 Service abstraction
`aipotluck/installer/service/base.py` defines:
```python
class ServiceManager(ABC):
    def install(self, exec_path: str, args: list[str], name: str, user_scope: bool) -> None: ...
    def uninstall(self, name: str) -> None: ...
    def start(self, name: str) -> None: ...
    def stop(self, name: str) -> None: ...
    def status(self, name: str) -> ServiceStatus: ...
```
`install.py` picks the concrete implementation via `platform.system()`.

### 4.2 Linux — systemd user unit (default), system unit optional
- Default: `~/.config/systemd/user/aipotluck.service`, enabled with
  `systemctl --user enable --now aipotluck.service`. No root needed. Starts
  at user login (or with lingering enabled via
  `loginctl enable-linger $USER` for boot-time start without login — call
  this out explicitly, don't silently enable linger).
- `--system` flag: writes `/etc/systemd/system/aipotluck.service`, needs
  sudo, runs at boot regardless of login — this is the "real" system service
  case the user mentioned.
- Unit just runs `python3 <path>/aipotluck_service.py`; `Restart=on-failure`.

### 4.3 macOS — launchd
- Default (user/agent scope): plist at
  `~/Library/LaunchAgents/com.aipotluck.service.plist`, loaded via
  `launchctl load -w`. Starts at login.
- `--system` flag: `LaunchDaemons` in `/Library/LaunchDaemons/`, needs sudo,
  starts at boot without login — the daemon-scope equivalent of a system
  service.
- `ProgramArguments` invokes the same python entrypoint; `KeepAlive` +
  `StandardOutPath`/`StandardErrorPath` for logs.

### 4.4 Windows — real Windows Service via `pywin32`, with a no-admin fallback
- True Windows Services need admin + a service wrapper. Use `pywin32`
  (`win32serviceutil`) — it's the standard, well-trodden path, pure Python,
  no extra native installer needed beyond the pip package.
- `pip install pywin32` as a dependency of the service component only
  (Linux/macOS installs don't need it).
- Default (no admin available / user declines UAC): fall back to a
  **Scheduled Task** (`schtasks /create /sc onlogon ...`) running the same
  script — no admin required, starts at logon, good enough for the "blank"
  phase.
- `--system` flag (requires elevation): registers as an actual Windows
  Service (`python aipotluck_service.py install` via pywin32's
  `win32serviceutil.HandleCommandLine`, then `sc config ... start= auto`) —
  runs at boot, no login required. This is the true system-service parity
  with systemd `--system` / launchd `LaunchDaemons`.

### 4.5 The "blank" service payload itself
`aipotluck/service/aipotluck_service.py`: minimal, dependency-free (stdlib only) for
now:
- A tiny loop / `http.server` health endpoint (`GET /healthz` → `200 ok`) so
  we have something externally observable and testable across all three
  service backends without needing the transport layer yet.
- Reads `runtime.json` written by the installer so it already knows where
  llama-server lives, in preparation for later orchestration duties, but
  does not start/stop llama-server yet in this phase — that wiring comes
  with the transport layer.
- Structured logging to the per-OS log dir from section 3.3.

## 4.5 The "always running" supervision layer (implemented)

`aipotluck/service/llama_supervisor.py` implements `LlamaSupervisor`, used by
`aipotluck/service/aipotluck_service.py`:

- On service start, spawns `llama-server` **in router mode** (CUR-1965) with args built from
  `runtime.json` (`--host`, `--port`, `--models-preset`, `--models-max`, `--gpu-layers`) -- no
  `-hf`/`--model` of its own. Passing no model puts llama-server into its own native multi-model
  router (confirmed against `vendor/llama.cpp/tools/server/server-models.cpp`): it auto-discovers
  every model already in the HF cache and routes each request by the `"model"` field in its body,
  loading/unloading instances on demand. This is what makes "the user switched models" something
  llama-server itself detects and handles, not something this project intercepts or proxies. See
  section 4.5.1 below for how a model's `ctx_size`/`parallel`/`cache_type_k`/`-v` get set.
- Polls `GET /health` on llama-server every 5s (`HEALTH_POLL_INTERVAL_SECONDS`) -- router-level,
  not per-model, so this stays meaningful even with zero models currently loaded.
- On crash (process exit), restarts with exponential backoff
  (1/2/5/10/20/30/60s), resetting to the start of the schedule if the
  process had stayed up 2+ minutes (`STABLE_UPTIME_RESET_SECONDS`) --
  avoids hammering restarts on a persistently broken config while still
  recovering fast from transient crashes.
- On service shutdown (SIGTERM/SIGINT), terminates llama-server gracefully
  (SIGTERM, 15s grace period, then SIGKILL) before exiting.
- Exposes live state via `LlamaProcessInfo` (pid, running, healthy,
  restart_count, last_exit_code) surfaced at `GET /status` on the aipotluck
  service's own HTTP endpoint (port 8069). A third endpoint, `GET
  /capabilities` (`aipotluck/diagnostics.py`), reports a full system/capability
  fingerprint built from the same detection code the installer itself uses
  (`platform_detect`/`build_strategy`/`build_cache`/`source_build`) -- OS,
  arch, glibc, CPU/memory/disk, GPU backend + CUDA compute capability, every
  build tool, and the install strategy this device would resolve to right
  now vs. what's actually installed. See section 3.2.1 above.
- `GET /status` and `/capabilities` both also carry `runtime_params`
  (`diagnostics.runtime_params`, one real function, not a copy per endpoint):
  router-level params (`host`/`port`/`gpu_layers`/`models_max`) plus, per model, the
  `ctx_size`/`parallel`/`cache_type_k`/`cache_type_v` this project computed for it and -- for each
  of those -- a plain-sentence reason, read straight off the `--models-preset` INI file and its
  sibling tuning JSON (`aipotluck.installer.model_presets`). Router mode means there's no single
  "active model" to summarize -- every sized model is reported, keyed by id.
  `aipotluck-local-client status` prints the same dict. See CLAUDE.md's "Runtime parameters"
  section for why this exists as a standing convention, not a one-off for the ctx_size/parallel
  case that prompted it (CUR-1965).

Two-tier self-healing, confirmed by test:
1. systemd `Restart=always` / launchd `KeepAlive` / Windows Service restarts
   `aipotluck_service.py` itself if it dies.
2. `LlamaSupervisor` inside that process restarts just `llama-server` if
   *it* dies, without needing to restart the whole Python service.

Verified end-to-end on this repo's Linux/arm64 dev host: real model
download via llama-server's own `-hf` fetcher (bartowski/Qwen2.5-0.5B
default placeholder), health-check transition to healthy, live inference
request, `kill -9` on the llama-server child recovered automatically
(new pid, `restart_count` incremented, healthy again ~35s later), and
clean shutdown terminates both processes with no orphans.

### 4.5.1 Router mode + automatic per-model sizing (`aipotluck/installer/model_sizing.py`/`model_presets.py`, CUR-1965)

CUR-1965 started as a single-model bug (a device silently running with 4x more KV-cache-eating
parallel slots than a single-user box needs) and its fix evolved into a real architecture change:
rather than this project detecting "the user switched models" itself (which would mean intercepting
or proxying every inference request), llama-server's own **router mode** already does exactly that
-- confirmed by reading `vendor/llama.cpp/tools/server/server-models.cpp` directly rather than
building a proxy first and discovering this afterward. So the service runs the router
(section 4.5), and this project's job narrows to making sure a model has a *correct* sizing preset
waiting for it before the router ever loads it.

**The preset file.** `--models-preset` is an INI file (`aipotluck/installer/model_presets.py`
owns its read/write, confirmed against `vendor/llama.cpp/common/preset.cpp`'s real PEG-grammar
parser): one section per model, keyed by the exact `repo:quant` identity `--cache-list`/
`model_pull.list_cached_models` already use (verified via
`common_preset_context::load_from_cache()`, which names each cache entry the same
`model.to_string()` call `--cache-list` prints -- so there's no separate identity-matching logic to
get wrong). Keys are llama.cpp's own long-form CLI flag spelling with dashes stripped
(`ctx-size`, `parallel`, `cache-type-k`, `cache-type-v`). Every other section -- another model, the
`[*]` global section, anything a person hand-edited -- is preserved verbatim on every write (always
read-modify-write, never regenerate-from-scratch). A sibling `<presets>.tuning.json` file (also
`model_presets.py`) carries the `{param: "reason string"}` traceability map per model -- kept
separate from the INI because llama-server's own preset grammar never needs to read it; it exists
purely for `status`/`/capabilities` (section 4.5 above).

**When a preset gets computed** (`model_sizing.ensure_preset`, called from three places):
1. `aipotluck-local-client pull` (`cli.py`'s `run_pull_model`) -- always recomputes
   (`force=True`); a deliberate re-pull is a reasonable time to re-size. After writing the preset,
   it best-effort asks a running router to `GET /models?reload=1` (`_reload_router_models`) so the
   new model + sizing are live immediately, rather than restarting the whole service the way
   `login`/`logout` do -- a real UX improvement router mode makes possible, since the router
   itself doesn't need restarting to pick up a newly-cached model.
2. `install.py`'s `run_install`, for the default `--model-hf`/`--model-path` -- pre-fetches (HF
   only; a `--model-path` file is already local) and pre-sizes it, so a fresh install already has
   a correctly-sized preset waiting instead of falling back to llama-server's stock defaults until
   the person runs `pull` themselves. `--no-model-pull` skips this for CI/dry-run contexts.
3. `runner.py`'s `backfill_missing_presets`, once on every service start, for any cached model
   that doesn't have a preset yet (`force=False` -- fills gaps, doesn't redo work). Recovers a
   deleted/hand-edited-away presets file, or a model that reached the cache some other way, without
   needing another explicit `pull`. It runs as a background job (visible in `GET /jobs`), after
   the HTTP server and the router are already up, because each probe loads a model and `status`
   has to answer from the moment the service starts; the router reloads when it finishes. Until
   then an unsized model is served on llama-server's own defaults.

All three call the same function, so there is exactly one place a sizing decision gets persisted --
never three call sites each writing their own copy that could drift from the others.

**The probe** (`model_sizing.probe_model_profile`): a short second `llama-server` spawn (fast --
the model's already cached at this point, so this is a local load, not a network fetch) at a small
probe context, just far enough to report its own hyperparameters by parsing its `print_info:`
startup log -- the same values a real `llama_model::print_info()` call prints, not re-derived from
the GGUF file independently. Reads `n_ctx_train` (trained context length), the per-layer KV-cache
footprint (`n_embd_k_gqa`/`n_embd_v_gqa`, `n_embd_head_k` for the quantization-block-size
divisibility check), and the model's own file size (`load_tensors: file size = ... MiB/GiB`, also
parsed from the log).

**The sizing math itself** (`compute_sizing`): KV-cache bytes/token = `n_layer * (n_embd_k_gqa *
bytes_per_element(cache_type_k) + n_embd_v_gqa * bytes_per_element(cache_type_v))` -- the same
formula CUR-1965 hand-derived and confirmed against a real device (~112 KiB/token for Llama 3.2 3B
at f16: 28 layers, 8 KV-heads, 128 head-dim). The budget is **80% of this device's TOTAL installed
RAM** (`_MEMORY_BUDGET_FRACTION`, matching Ollama's own `freeMemory*80/100` eviction threshold),
**not a live "available right now" reading** -- a deliberate choice: background memory headroom
fluctuates over a device's uptime, and a value picked once from a single live snapshot would bake
in whatever happened to be true at that moment, which is exactly the "it just works until it
doesn't" gap CLAUDE.md's "Runtime parameters" section warns about. The model's own file size (from
the probe) is subtracted from that budget before sizing context, since router mode may hold this
model loaded for a long time regardless of what else the box is doing at sizing time. **Future
improvement, deliberately not attempted here:** monitor a model's *actually observed* memory
headroom over its running lifetime and adjust from there if it drifts from this static budget.

`parallel` is always fixed at `1` -- a single-user local device gets nothing from llama-server's
default of 4 slots except a 4x-smaller usable context for the same memory, and it keeps
`--models-max 1`'s own "one model, the whole budget" assumption valid (see section 4.5.2 below --
raising `models_max` is a real capacity tradeoff this project's memory math doesn't account for).
KV cache type prefers `q8_0`, falling back to `q4_0` only when headroom is too tight for `q8_0` to
reach the model's full trained context, and skipping quantization entirely (`cache_type_k`/`v`
left unset, llama-server's f16 default applies) when the model's head dimension doesn't divide
evenly into the 32-element quantization block size -- confirmed against
`vendor/llama.cpp/src/llama-context.cpp`'s own startup validation, which refuses to start rather
than silently falling back itself.

A sizing failure (the probe times out, a required hparam is missing from the log, total memory
can't be measured on this platform) raises `ModelSizingError`, which every one of the three call
sites above catches and logs as a warning -- never fatal to the pull/install/service-start it's
part of, and never overwrites an existing preset with a guess.

### 4.5.2 `--models-max` and memory budgeting

`--models-max` (default `1`, `aipotluck.service.runner.DEFAULT_MODELS_MAX`) caps how many models
the router may hold loaded simultaneously. The sizing math in 4.5.1 above assumes *one* model gets
the whole memory budget -- raising this is a real capacity tradeoff (multiple models sharing one
memory budget, none of which this project currently re-sizes to account for the others being
loaded too), not a free win, and isn't accounted for anywhere in this repo today.

## 4.6 Portability refactor: one service payload, three OS wrappers

To maximize code reuse across Linux/macOS/Windows, the service logic was
split into a layered, OS-agnostic core plus thin OS-specific entry points:

```
aipotluck/service/llama_supervisor.py   -- OS-agnostic: subprocess.Popen + polling (stdlib only)
aipotluck/service/runner.py             -- OS-agnostic: AipotluckServiceRunner
                                  (HTTP status server + supervisor lifecycle,
                                  exposes start()/stop()/wait(), no signal
                                  handling, no service-framework imports)
aipotluck/service/aipotluck_service.py  -- thin CLI wrapper: argparse + signal handlers
                                  around AipotluckServiceRunner. This exact
                                  script is what systemd, launchd, AND the
                                  Windows Scheduled Task (no-admin default)
                                  all invoke identically.
aipotluck/service/windows_service_host.py -- pywin32 ServiceFramework wrapper around
                                  the same AipotluckServiceRunner, used only
                                  for the Windows --system path (a true
                                  Windows Service needs SCM callbacks
                                  (SvcDoRun/SvcStop) instead of Unix signals)
```

This means ~95% of the runtime logic (HTTP status endpoint, llama-server
spawn/health-poll/restart-with-backoff, clean shutdown) is identical
bytecode across all three OSes and all four service-registration
mechanisms (systemd unit, launchd plist, Scheduled Task, Windows Service).
Only the "how does this OS keep the wrapping process alive" layer differs,
which is inherently OS-specific and was already isolated in
`aipotluck/installer/service/{systemd,launchd,windows_service}.py`.

A real bug was caught and fixed during the original Linux implementation
that motivated splitting the HTTP server onto its own thread in
`runner.py`: calling `http.server`'s `shutdown()` synchronously from a
signal handler running on the same thread as `serve_forever()` deadlocks
(the shutdown call blocks waiting for the serve loop to notice, but the
serve loop can't proceed until the signal handler returns). Moving the
HTTP server to its own thread means `stop()` can call `shutdown()`
directly and safely from any thread (a signal handler, a pywin32 SvcStop
callback, whatever) without that hazard.

## 4.7 Windows: bootstrapping Python itself

Explicit product requirement: **the Windows installer must check for
Python and install it if missing**, since unlike Linux/macOS, a stock
Windows machine commonly has no Python at all, and the installer itself is
Python.

Two cooperating pieces:

- `aipotluck/installer/python_bootstrap.py` (`find_python()` / `ensure_python()`):
  pure-Python, cross-platform version-checking logic. On Windows, if no
  interpreter >= 3.9 is found, it shells out to
  `winget install --id Python.Python.3.12 -e --silent
  --accept-package-agreements --accept-source-agreements`, then re-resolves
  by probing `%LOCALAPPDATA%\Programs\Python\` and `%ProgramFiles%`
  directly (winget does not refresh the current process's PATH). Raises a
  clear, actionable `PythonNotFoundError` with manual-install instructions
  if winget itself is unavailable or fails. On macOS/Linux, `ensure_python`
  only detects and raises an actionable brew/apt/dnf message -- no
  sudo-gated auto-install on those platforms, since they ship Python by
  default in practice.
- `packaging/windows/install.ps1`: the *true* zero-prerequisite Windows
  entry point, since `python -m aipotluck.installer.install` can't run at all
  without Python already present. Plain PowerShell (no dependencies),
  mirrors the exact same find-or-winget-install logic, then hands off to
  `python -m aipotluck.installer.install` with equivalent flags. This is what a
  Windows user (or a future signed .exe wrapper) actually runs first.

`aipotluck/installer/install.py`'s service-registration step also calls
`ensure_python()` itself (not just the PS1 script) before writing the
service's `ExecStart`/`ProgramArguments`/schtasks command line -- so the
service always points at a *validated* interpreter path, not a blindly
trusted `sys.executable`, which matters most on Windows where the
resolved python.exe may have just been installed by winget in this same
run.

STUB STATUS for this whole section: written against documented winget
CLI behavior (`winget install`, silent flags, package ID
`Python.Python.3.12`) and pywin32's public API surface, but **not
exercised on real Windows hardware** -- no Windows host is available in
this environment. Everything downstream of `ensure_python()` returning a
valid interpreter path is the exact same code already tested end-to-end on
Linux.

## 4.8 Public one-line installer layer

Two thin bootstrap scripts at the repo root, meant to be posted publicly
and piped into a shell (`curl ... | bash`, `irm ... | iex`):

- `install.sh` (Linux/macOS): checks for `git`, clones (or updates an
  existing checkout of) the repo into `~/.aipotluck/src`, then execs
  `python3 -m aipotluck.installer.install` with any passed-through arguments. Python
  presence is not checked here -- that's `aipotluck/installer/install.py`'s job via
  `python_bootstrap.ensure_python()`, which already has the right
  per-OS messaging.
- `install.ps1` (Windows, repo root -- distinct from
  `packaging/windows/install.ps1`): checks for `git` (installing it via
  winget if missing, mirroring the Python auto-install pattern), clones
  into `%LOCALAPPDATA%\aipotluck\src`, then hands off to
  `packaging/windows/install.ps1` (the one that finds-or-installs Python).

Both scripts default to the real public clone URL
(`https://github.com/currentai-org/aitpotluck-local-client.git`), overridable via
`AIPOTLUCK_REPO_URL` for a fork or a local mirror -- see README.md's "One-line
public installer" section for exact end-user usage.

No installer logic is duplicated in either bootstrap script -- both are
strictly "get the source, then call the already-tested installer,"
consistent with every other layer in this project reusing rather than
reimplementing.

Verified end-to-end on this Linux/arm64 dev host using a local bare git
repo as a stand-in "public" remote (`file://` URL via
`AIPOTLUCK_REPO_URL`): fresh clone with submodule init, re-run against an
existing checkout (updates via fetch + reset --hard rather than
re-cloning), and the placeholder-URL guard correctly refusing to run
without an override. The Windows `install.ps1` (repo root) is untested
like the rest of the Windows surface (no Windows host available).

## 5. Open questions before implementation

Resolved during implementation:

1. Default install scope: **per-user (no admin), autostart at login** --
   implemented as the default; `--system` is opt-in on all three OS
   backends.
3. GPU backend default: **auto-detect with CPU fallback** -- implemented in
   `platform_detect.detect_gpu_backend`; unknown/unsupported asset
   combinations (e.g. no linux-arm64-cuda release) fall back to the CPU
   asset automatically (`fetch.resolve_asset`).
4. Installer language: **stdlib Python**, confirmed sufficient; no native
   wrapper needed yet.
5. Default model: **no bundled model** -- defaults to a small placeholder
   (`bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M`) fetched via
   llama-server's own `-hf` flag so the service has something to supervise
   out of the box; `--model-hf` / `--model-path` override for real use.

Still open:

2. Should the installer also do a foreground smoke-test run of
   `llama-server` before handing off to the service, separate from the
   supervised background process? (Not implemented -- the supervisor's own
   health-poll-until-healthy on startup serves this purpose today.)

## 6. Non-goals for this phase

- No custom transport layer yet (explicitly deferred per instructions).
- No GUI installer wizard yet (packaging/ is a placeholder).
- No auto-update mechanism yet.
- No support for building llama.cpp from source by default (submodule is
  reference/fallback only, not the default install path).
