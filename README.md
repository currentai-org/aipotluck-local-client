# aipotluck-local-client

Cross-platform installer and local inference server wrapper. Wraps
[llama.cpp](https://github.com/ggml-org/llama.cpp) (pinned as a git
submodule at `vendor/llama.cpp`) and installs a small companion Python
service that keeps `llama-server` running at all times.

The installer prefers a prebuilt release binary, but not every host has one it
can actually run: `aipotluck/installer/build_strategy.py` decides this per
host (no published asset for the detected backend/arch, e.g. every
Jetson-class arm64+CUDA board; or the best available asset needs a newer
glibc than the host has), and `aipotluck/installer/source_build.py` builds
`llama-server` from the same pinned `vendor/llama.cpp` tag when it's needed.
See "Confirmed hardware" below for what's actually been run where.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design rationale.

## Confirmed hardware

| Host | OS | Backend | Install path | Status |
| --- | --- | --- | --- | --- |
| x86_64 laptop | Ubuntu 24.04 | CPU | prebuilt `linux-x64-cpu` asset | confirmed |
| NVIDIA Jetson Orin Nano 8GB (Developer Kit Super) | JetPack 6.2.3 (Ubuntu 22.04, glibc 2.35) | CUDA (SM 8.7) | our own binary cache (`linux-arm64-cuda-sm87`) | confirmed -- cache-hit installs in ~5s |
| NVIDIA Jetson Orin NX 16GB (Developer Kit) | JetPack 6.x / L4T R36.5.0 (Ubuntu 22.04, glibc 2.35) | CUDA (SM 8.7) | our own binary cache (`linux-arm64-cuda-sm87`) -- same cache entry as the Orin Nano above, since both report the same compute capability | confirmed -- cache-hit installs, real GPU-accelerated inference verified |

Everything else in "Status" just below is implemented against documented OS
conventions but not run on real hardware yet.

## Status

- Linux: implemented and tested (systemd --user service backend, live
  llama-server supervision with crash-restart, verified end-to-end with a
  real model download and inference request).
- macOS: implemented per Apple's documented launchd/plist conventions,
  reusing the identical, already-tested service payload. **Untested** (no
  macOS host available in this environment). See
  `aipotluck/installer/service/launchd.py`, `packaging/macos/README.md`.
- Windows: implemented per documented winget/schtasks/pywin32 conventions,
  including a PowerShell bootstrapper (`packaging/windows/install.ps1`)
  that installs Python itself via winget if none is found, plus a real
  pywin32 Windows Service host for the `--system` path. **Untested** (no
  Windows host available). See `aipotluck/installer/service/windows_service.py`,
  `aipotluck/service/windows_service_host.py`, `aipotluck/installer/python_bootstrap.py`,
  `packaging/windows/README.md`.

All three OS backends drive the exact same portable service payload
(`aipotluck/service/runner.py` -> `aipotluck/service/llama_supervisor.py`) --
only the OS-native "keep this process running" registration differs
(systemd unit / launchd plist / Scheduled Task or Windows Service).

## Quick start

Linux / macOS:
```bash
git clone https://github.com/currentai-org/aitpotluck-local-client.git
cd aitpotluck-local-client
python3 -m aipotluck.installer.install --backend auto
```

Windows (no prerequisites, installs Python itself if missing):
```powershell
git clone https://github.com/currentai-org/aitpotluck-local-client.git
cd aitpotluck-local-client
powershell -ExecutionPolicy Bypass -File packaging\windows\install.ps1
```

No `--recurse-submodules`: the only submodule is `vendor/llama.cpp` (~174MB), and it is needed
only on a host that has to **compile** llama.cpp. Most don't -- they resolve to an upstream
release asset or one of our own cached custom builds and never build anything. When a source
build really is required, `source_build._init_vendor_submodule` fetches the source then, shallowly.
Cloning with `--recurse-submodules` anyway is harmless, just a download most people won't use.

## One-line public installer

`install.sh` (Linux/macOS) and `install.ps1` (Windows) at the repo root are
thin bootstrap wrappers meant to be posted publicly and piped straight into
a shell -- no manual `git clone` step required:

```bash
# Linux / macOS -- standard practice for this kind of installer: no arguments needed.
curl -fsSL https://raw.githubusercontent.com/currentai-org/aitpotluck-local-client/main/install.sh | bash

# advanced use only -- installer flags, passed straight through:
curl -fsSL https://raw.githubusercontent.com/currentai-org/aitpotluck-local-client/main/install.sh | bash -s -- --backend cuda --model-hf "org/repo:Q4_K_M"

# pointing at a fork/branch instead of the default: the env vars have to go on bash's side of the
# pipe, not curl's -- a leading assignment on a pipeline only applies to the first command in it
# (curl here), so putting them before curl silently does nothing (confirmed live, CUR-1965).
curl -fsSL https://raw.githubusercontent.com/currentai-org/aitpotluck-local-client/main/install.sh | \
  AIPOTLUCK_REPO_URL=https://github.com/<owner>/<fork>.git AIPOTLUCK_REF=main bash
```

```powershell
# Windows, no arguments:
irm https://raw.githubusercontent.com/currentai-org/aitpotluck-local-client/main/install.ps1 | iex

# Windows, with arguments (irm|iex can't take params directly, use this form):
$script = irm https://raw.githubusercontent.com/currentai-org/aitpotluck-local-client/main/install.ps1
Invoke-Expression "& { $script } -Backend cuda -System"
```

Both scripts assume nothing is installed yet and do their best to fix that
themselves: `install.sh` checks for `git`/`python3` and, if either is
missing, attempts a non-interactive install via whatever package manager it
finds (`apt-get`/`dnf`/`yum`/`pacman`/`apk`/`zypper`/`brew`) before falling
back to a clear, actionable error; `install.ps1` installs `git` via winget
first on Windows if it's missing. Both then clone the repo (into
`~/.aipotluck/src` / `%LOCALAPPDATA%\aipotluck\src`) and hand off
immediately to the already-implemented, already-tested installer
(`aipotluck/installer/install.py` on Linux/macOS, `packaging/windows/install.ps1`
on Windows, which itself finds-or-installs Python before calling
`aipotluck.installer.install`). No installer logic is duplicated in the
one-liners themselves. Re-running either script updates the existing
checkout (`git fetch` + `reset --hard`) rather than re-cloning.

**The install always finishes in a "logged out" state** -- no Pangolin
credentials, and the service holds `llama-server` and the tunnel back until
you pair the device. See "Pairing" below.

## Installer options

The Python installer's flags (the PowerShell wrapper exposes the equivalent
`-System`/`-Backend`/`-ModelHf`/`-Tag`/`-NoStart`/`-Verbose` flags):

```
--tag TAG             Override the pinned llama.cpp release tag
--backend {auto,cpu,cuda,vulkan,rocm}
--install-dir PATH    Override the default per-OS install root
--system              Install machine-wide (requires elevation)
--no-start            Install the service but don't start it
--no-service          Only fetch/extract llama.cpp, skip the Python service
--model-hf REPO:QUANT  Pre-fetch + pre-size this model as the device's default
                       (default: a small Qwen2.5-0.5B placeholder). llama-server
                       runs in router mode (see "Models: pull / list" below) and
                       can serve any model already downloaded, not only this one.
--model-path PATH      Pre-size a local GGUF file instead of fetching one
--no-model-pull        Skip the pre-fetch/pre-size step (CI/dry-run contexts)
--models-max N         Models the router may hold loaded at once (default: 1)
--gpu-layers VALUE     llama-server -ngl value: int, 'auto', or 'all' (applies
                       to every model the router loads, not just the default one)
--server-host / --server-port  llama-server bind address (default 127.0.0.1:8080)
--no-source-build      Fail instead of building llama-server from source when
                       no prebuilt asset is viable for this host
--jobs N               Parallel build jobs for a from-source build (default:
                       auto, capped by available RAM)
--build-timeout SECS   Wall-clock ceiling for a from-source build (default: 5400)
--allow-apt-install    Consent up front to installing missing build deps via
                       'sudo apt-get install' (skips the interactive y/N prompt)
--no-apt-install       Never offer to auto-install build deps, even interactively
-v / --verbose
```

### Building from source (and our own binary cache)

Not every host has a prebuilt release binary it can actually run --
`build_strategy.py` checks this (not just "does an asset exist for this
os/arch/backend", but "does the *specific* asset resolve to something this
glibc can run"). When it doesn't, two things are tried in order before
anything gets compiled:

1. **Our own cache of prebuilt binaries** (`build_cache.py`,
   `llama_custom_builds.json`) -- once we've built a binary for a given
   host/backend/GPU-architecture combination once, there's no reason to pay
   a 30-90 minute compile again on every matching device. If a cached entry
   matches, it's downloaded (cached, checksum-verified, same as any other
   asset) and compilation is skipped entirely.
2. **`source_build.py`** builds `llama-server` from the pinned
   `vendor/llama.cpp` tag when neither an upstream asset nor a cached custom
   one matches.

This is all automatic; you'll see it happen if the installer logs "No
viable upstream asset for this host... but a cached custom build exists" (no
compile) or "No viable prebuilt asset for this host... building llama-server
from source" (a real compile) instead of "Resolved asset:".

Two concrete cases this covers today:

- **No prebuilt asset for the detected backend at all** -- every
  Jetson-class arm64+CUDA board: `nvidia-smi` genuinely works there and CUDA
  is real, but llama.cpp's release CI has never published a
  `linux-arm64-cuda` asset.
- **The best available asset needs a newer glibc than the host has** --
  llama.cpp's release CI builds its Linux arm64 assets against a newer base
  image than most arm64 boards actually run. Confirmed live: the pinned
  `linux-arm64-cpu` asset needs glibc 2.38; JetPack 6.2.3 (Ubuntu 22.04)
  ships 2.35, so even the CPU-only fallback refuses to start there.

A source build needs `cmake`, `git`, a C++17 compiler, OpenSSL dev headers,
and the OpenMP runtime -- plus, per backend, a GPU vendor toolchain: `nvcc`
for CUDA, `glslc` + Vulkan headers for Vulkan, `hipcc` for ROCm. This list
isn't guessed from one host's missing package -- it's cross-referenced
against every Linux job in llama.cpp's own release CI (cpu/cuda/vulkan/rocm),
intersected with what a native build of just the `llama-server` target
actually needs (their CI installs several things -- `ninja-build`,
`python3-venv`, `git-lfs`, `libjpeg-dev` -- for its own portable-build/test
concerns that don't apply here; confirmed `libjpeg-dev` isn't even a real
llama.cpp dependency, since image loading goes through the header-only
`stb_image` instead of libjpeg). See `source_build.py`'s module docstring for
the full evidence trail.

For everything except a GPU vendor toolchain, the installer can install the
gap itself via `sudo apt-get install` -- but only with your explicit
consent: pass `--allow-apt-install` up front, or answer yes to the
interactive prompt it shows otherwise (that prompt never appears, and nothing
is auto-installed, without a real terminal to ask through -- e.g. the public
`curl | bash` one-liner, which has no stdin a person could answer through).
`sudo`'s own password prompt is untouched either way; this installer only
ever supplies the package list, never a password. `--no-apt-install` turns
this off entirely and goes back to just printing the exact `apt-get install`
line for whatever's missing. A GPU vendor toolchain is never auto-installed
under any of these flags -- each is a multi-GB, distro/vendor-specific
install (on Jetson/JetPack, CUDA specifically is the vendor-managed OS image,
not something this installer should touch); a missing `nvcc`/`glslc`/`hipcc`
always stays an instruction.

**Two easy-to-miss gaps if you install these yourself ahead of time, both the
same shape:** a plain, non-`REQUIRED` `find_package(...)` in llama.cpp's own
CMake that degrades **silently** -- configure still succeeds -- rather than
failing loud.

- HTTPS support (used by `-hf` and `--cache-list` -- i.e. this project's own
  `pull`/`list` commands) needs OpenSSL dev headers (`libssl-dev` on
  Debian/Ubuntu). Missing them doesn't fail the build; it only fails later,
  at runtime, the first time something tries to download a model.
- OpenMP multi-threading on the CPU backend needs the OpenMP runtime
  (`libgomp1` on Debian/Ubuntu). Missing it doesn't fail the build either --
  just a `message(WARNING "OpenMP not found")` buried in build output, and
  the binary silently falls back to single-threaded CPU inference, no error,
  just much slower than it should be.

The installer's preflight check catches both before ever starting a build
that can otherwise take well over an hour on a low-power board.

A from-source build is slow and memory-hungry (`--jobs` defaults to a
conservative estimate based on available RAM, since a naive `-j$(nproc)` can
OOM or swap-thrash a low-memory board) and needs real disk space (a few GB
for the build; the installer refuses up front if less than 6GB is free
rather than fail 40 minutes in from `ENOSPC`). Re-running the installer
after a successful source build is fast -- it's keyed to the exact
tag+backend combination and skips straight to reusing the binary already
built.

#### Custom binary cache (`llama_custom_builds.json`)

This is a separate, optional manifest alongside `llama_version.json`, for
binaries **we've** built and hosted ourselves -- not upstream releases. Same
shape (`release_base_url` + `tag` + `{key: {file, sha256}}`), same
download/verify/extract code path (`fetch.py`), just a different source and
a different key format: `{os}-{arch}-{backend}`, plus `-sm{N}` for CUDA
(e.g. `linux-arm64-cuda-sm87` for a Jetson Orin) -- since different
Jetson-class generations (Xavier SM 7.2, Orin SM 8.7, Thor SM 10.x) are not
interchangeable; a binary built for one `CMAKE_CUDA_ARCHITECTURES` value
will not run correctly targeting a different one.

The file is entirely optional -- a checkout without one (or with an empty
`assets` map) just falls through to a real source build every time, exactly
as if this feature didn't exist.

To add an entry after building on a new host, `scripts/package_custom_build.py`
is the standardized tool for this (written up front to be reused, not a
one-off) -- run it right after the installer finishes a real source build
there:

```bash
python3 scripts/package_custom_build.py --build-dir /path/to/llama.cpp-build
```

It stages the build's `bin/` directory into a flat top-level `llama-<tag>/`
directory (matching upstream's own release archive layout exactly --
`llama-b10989/llama-server`, `llama-b10989/libggml-cuda.so.0`, etc., no
nested `bin/` -- confirmed against a real llama.cpp release tarball), then
**verifies relocatability for real**: copies the staged directory to a fresh
path elsewhere and actually runs `llama-server --list-devices` from there,
refusing to package anything that fails (a real safeguard, not a formality
-- this exact project shipped a non-relocatable build once, and that failure
mode is a dynamic-linker error at runtime, not something a "did cmake
configure OK" check would ever catch). Tag and cache key both default to
sensible values (the pinned tag from `llama_version.json`; the key
auto-derived from *this host's own* detected profile via the same code
`build_cache.py` uses to look one up) -- override with `--tag`/`--key` only
when packaging on behalf of a different host than the one you're running on.

It prints the exact next two steps: the `gh release upload`/`create`
command, and the JSON snippet to add to `llama_custom_builds.json` (file
name + sha256, computed for you) -- both ready to copy-paste. Uploading the
release asset is a real publish to this project's public repo, so that step
is left as something a person runs deliberately, not something the script
does on its own.

After install, the service:

- Exposes a health check at `http://127.0.0.1:8069/healthz`, detailed
  status (including live llama-server supervisor state) at
  `http://127.0.0.1:8069/status`, and a full system/capability fingerprint
  at `http://127.0.0.1:8069/capabilities` -- OS/distro, arch, glibc
  version, CPU/memory/disk, GPU backend + CUDA compute capability, every
  build tool `source_build.py` checks for, and the actual install strategy
  this device would resolve to right now (upstream asset vs. our own binary
  cache vs. a real source build, and why) alongside what's *currently*
  installed. Built from the exact same detection code the installer itself
  uses (`aipotluck/diagnostics.py`), so this is a live, remotely-queryable
  answer to "what would happen if I reinstalled this right now" -- useful
  for diagnosing a device that's already in a broken state, since every
  section degrades independently (`{"error": ...}`) rather than the whole
  endpoint failing if one probe does.
- **Every runtime parameter this project computes automatically is traceable,
  locally and remotely, from one real source** (`aipotluck.diagnostics.runtime_params`)
  -- both `/status` and `/capabilities` carry a `runtime_params` key (the
  llama-server flags actually in effect, plus a plain-sentence reason for
  anything computed rather than explicitly set, via `runtime.json`'s
  `llama_cpp.tuning`), and `aipotluck-local-client status` prints the same
  data. Standing convention, not a one-off -- see CLAUDE.md's "Runtime
  parameters" section for why.
- **Actively supervises `llama-server`**: starts it on service startup,
  polls `/health` every 5s, and restarts it automatically on crash with
  exponential backoff (1s, 2s, 5s, 10s, 20s, 30s, 60s -- resets if the
  process stayed up 2+ minutes). On service shutdown, llama-server is
  terminated cleanly (SIGTERM, then SIGKILL after 15s if unresponsive).
- Two-tier self-healing: systemd/launchd/Scheduled Task/Windows Service
  restarts the Python service process itself if it dies; the Python
  service's `LlamaSupervisor` restarts just `llama-server` if it crashes.
  Between the two, llama-server should stay running continuously.

`llama-server`'s own OpenAI-compatible API is reachable directly at
`http://127.0.0.1:8080` (or whatever `--server-host`/`--server-port` you set).

## Pairing: login / logout / status

The installer never asks for Pangolin credentials -- every install finishes
logged out, and the service (`aipotluck/service/runner.py`) won't start
`llama-server` or the tunnel (`newt`) until you log in. The installer drops
a small `aipotluck-local-client` wrapper onto your PATH (see "CLI on PATH"
below) that runs the thin login/logout/status CLI
(`aipotluck/installer/cli.py`) against whatever device is installed here:

```bash
aipotluck-local-client login     # prompts for the pairing JSON -- paste it, press Enter
aipotluck-local-client status    # asks the running service for its login/tunnel/llama-server state
aipotluck-local-client logout    # unpairs; llama-server and the tunnel stop until you log in again
```

Get the pairing JSON from **Settings -> Local Inference -> Add a managed server** on
aipotluck.org -- click "Pair", then its **Copy** button puts
`{"tunnelId": "...", "tunnelSecret": "...", "tunnelEndpoint": "..."}` on your clipboard (shown
once; it isn't retrievable again after that). Paste that straight into the `login` prompt. Two
alternatives to pasting: `login --credentials-file creds.json` reads the same JSON shape from a
file, and `login --tunnel-id X --tunnel-secret Y --tunnel-endpoint Z` accepts the three values as
separate flags for scripting (all three or none). All three forms are mutually exclusive, and
every flag here is given after the subcommand (`login --tunnel-id ...`), not before
(`--install-dir` and the other shared flags below are the same way; see cli.py's own comment on
why).

Logging in/out edits `runtime.json`'s `tunnel` section and `logged_in` flag,
downloads the pinned `newt` binary the first time (cached after that), and
restarts the already-installed service so it picks up the change -- it
never touches the llama.cpp install itself. If `aipotluck-local-client` isn't
on PATH yet (a fresh shell hasn't picked it up, or this was a `--no-service`
install, which skips the shim) fall back to
`python3 -m aipotluck.installer.cli login` from this checkout.

**`status`'s tunnel line distinguishes "the newt process is running" from "the tunnel is actually
connected"** (`tunnel_connected` in its output) -- these are genuinely different claims. `newt`
retries forever on its own and never exits just because it can't reach Pangolin, so a real
connection failure looks identical to success under `running: true` unless this is checked
explicitly (confirmed live: a device ran with `running: true` for 20+ hours while the tunnel was
never actually up, because Pangolin's endpoint was pointing at an address the device couldn't
reach).

The check (`aipotluck/service/newt_supervisor.py`) combines two signals, in priority order:

1. **newt's own `--health-file`** (newt >= 1.17.0) -- a real runtime signal newt computes from its
   actual connection state, not text matching. It writes the literal content `ok` once genuinely
   connected. **Confirmed live that this is not sufficient alone**: newt never refreshes it on
   later pings, and critically does *not* clear it if an already-established connection later
   drops -- killing a real, connected Pangolin server out from under a live newt process left the
   file saying `ok` while newt went straight into its own `ERROR:`-logged retry loop. A stale `ok`
   would read as healthy forever if trusted by itself.
2. **A tail of `newt.log`**, kept specifically to catch that gap -- newt logs a fresh `ERROR:` line
   on every failed retry for as long as it's genuinely still failing, so a recent `ERROR:` line
   overrides a stale `ok` health file. A log line that itself claims success is a fallback
   positive signal, not an independent source of truth.

`NewtSupervisor` also deletes any stale health file before every fresh spawn (belt-and-suspenders
on top of newt's own attempt to do the same at its startup), so a leftover `ok` from a previous
process is never inherited by a new one that hasn't reconnected yet.

`status` prints an explicit warning (and points at `newt.log`) when it sees `running: true` next
to a confirmed-disconnected tunnel; `tunnel_connected: null` (no signal yet from either check,
e.g. right after a fresh start) is left alone rather than treated as a failure.

## Models: pull / list

`llama-server` runs in its own **router mode** (CUR-1965): the service never pins a single
`-hf`/`--model` of its own. One always-running process auto-discovers every model already in the
Hugging Face cache and serves whichever one an inference request's own `"model"` field names,
loading/unloading instances on demand (`--models-max` caps how many at once -- `1` by default, see
below). This is what makes "the user switched models" something llama-server itself detects and
handles -- confirmed against `vendor/llama.cpp/tools/server/server-models.cpp` -- rather than
something this project has to watch for or proxy.

`pull` and `list` (`aipotluck/installer/model_pull.py`) exist to get a model downloaded and
correctly sized *before* it's first requested, rather than paying a cold download plus
stock-defaults sizing on that first real request:

```bash
aipotluck-local-client pull bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M   # download + size + speed-check
aipotluck-local-client list                                              # what's cached, with grades
aipotluck-local-client benchmark [model]                                 # re-measure speed only
```

`pull` accepts any Hugging Face `repo` or `repo:quant` target and hands it straight to
`llama-server`'s own `-hf` downloader rather than re-implementing Hugging Face's GGUF-resolution
logic (matching a quant string to the right file, split-GGUF handling, etc). It blocks until the
model has actually finished downloading *and* loading successfully (a real `/health` check on a
throwaway port, not just "the download finished"), sizes it (see below), then -- rather than
restarting the whole service the way `login`/`logout` do -- best-effort asks a running router to
`GET /models?reload=1` so the new model and its sizing are live immediately. Independent of login
state -- pulling a model doesn't need pairing.

`list` reads the same on-disk cache `-hf` writes into and `pull` reads from -- a real Hugging
Face Hub cache layout (`$LLAMA_CACHE` / `$HF_HUB_CACHE` / `$HUGGINGFACE_HUB_CACHE` /
`$HF_HOME/hub` / `$XDG_CACHE_HOME/huggingface/hub` / `~/.cache/huggingface/hub`, in that order --
see `vendor/llama.cpp/common/hf-cache.cpp`) -- via `llama-server`'s own `--cache-list` flag, for
the same "don't re-implement it" reason `pull` reuses `-hf`. Router mode means there's no single
"active" model anymore -- any cached model can be requested at any time -- so instead each row
carries its speed grade (below), plus a note when it has no sizing preset yet or has never been
benchmarked.

### Automatic runtime sizing (`aipotluck/installer/model_sizing.py`)

Every `pull` sizes the model it just downloaded -- `ctx_size`, `parallel`, and the KV cache type
-- and writes the result into `--models-preset`, an INI file llama-server's router reads and
applies per model (`aipotluck/installer/model_presets.py` owns that file's read/write). The same
sizing routine also runs once at service startup for any cached model that doesn't have a preset
yet (`runner.backfill_missing_presets`) -- recovers a deleted/hand-edited-away presets file, or a
model that reached the cache some other way, without needing another explicit `pull`.

To size a model, a short second `llama-server` spawn (fast -- the model's already cached, this is
a local load, not a network fetch) at a small probe context loads just far enough to report its
own hyperparameters: trained context length, per-layer KV-cache footprint, and file size. From
those, plus this device's **total installed RAM** (not a live "available right now" reading --
see below), it picks:

- **`ctx_size`** -- the largest context that fits the memory budget after the model's own weight
  size comes out of it, capped at the model's own trained context (there's no benefit to
  requesting more than that).
- **`parallel`** -- always `1`; llama-server's default of 4 slots divides the same memory budget
  four ways for no benefit on this single-user local device. It also keeps `--models-max 1`'s own
  memory math valid -- see "Router mode and memory budgeting" below.
- **`cache_type_k`/`cache_type_v`** -- `q8_0` if the model's head dimension supports a quantized
  KV cache and there's room to reach the full trained context with it; `q4_0` if headroom is
  tighter than that; left at llama-server's f16 default if the architecture can't use a quantized
  cache at all (a head dimension that doesn't divide evenly into the 32-element quantization block
  size -- llama.cpp itself refuses to start in that case, confirmed against its own startup
  validation).

**Why total RAM, not a live "available now" reading:** background memory headroom fluctuates over
a device's uptime, and a value picked once from a single live snapshot would bake in whatever
happened to be true at that moment. `_MEMORY_BUDGET_FRACTION` (80% of total RAM, matching Ollama's
own `freeMemory*80/100` eviction threshold) is the reserved headroom for the OS, other processes,
and compute/attention scratch buffers that scale with context but aren't captured by the KV-cache
formula alone -- an openly-approximate policy, not a precise measurement. **Future improvement,
deliberately not attempted here:** monitor a model's *actually observed* memory headroom over its
running lifetime and adjust sizing from there if it drifts from this static budget.

Every sizing decision is written alongside the preset, in a sibling `{param: "reason string"}`
JSON file (`model_presets.write_tuning`/`read_tuning`) -- visible from both
`aipotluck-local-client status` and `GET /capabilities`/`GET /status`'s `runtime_params`, per
CLAUDE.md's "Runtime parameters" convention. If sizing can't be done (memory can't be measured on
this platform yet, the probe times out, a required hparam can't be parsed, ...) it fails loud into
a log warning and leaves whatever preset that model already had (or none) untouched -- it never
blocks the download/pull itself, and never guesses.

### Speed grading (`aipotluck/installer/model_perf.py`)

Sizing answers "will this model *fit*". Grading answers the other question a small device makes
urgent: **will a reply actually arrive before the web app gives up?** On a CPU-only mini PC or a
Jetson it is easy to pull a model that loads perfectly, appears in the chat model picker, and then
never finishes a turn.

The budget is not a guess. `aipotluck.org`'s `deadlineLadder.ts` derives
`MAX_SAFE_GENERATION_TIMEOUT_MS = 155_000`, a local turn is clamped to exactly that, and the clock
starts *after* that app's prestream chain -- so cold model load, prefill and decode all come out of
it. Two more facts from that side shape the estimate: every reply is hard-capped at **1024 tokens**,
and the prompt is **never truncated** (a typical turn is ~2k tokens, 8-10k once retrieval or
web-search grounding fires, 32k+ with an attached file).

`pull` therefore benchmarks what it just downloaded, with `llama-bench` at the *same* K/V cache
types and GPU-layer count the preset will actually serve at, and fits per-token cost as a straight
line in KV-cache depth. The grade then answers the question users actually feel: **how much
thinking and answer arrives before the stream is cut off**. `N_out` is the number of output tokens
that fit in the budget, priced at a 32k-token context -- a long-but-plausible conversation, and the
same depth for every model so the numbers compare.

Pricing each model at its *own* full context instead is accurate and was tried: verified on a
Jetson, a 3B really does fall to 3 tok/s once 131k tokens are in its KV cache. But no conversation
gets there, and it quietly re-created the inversion this scheme exists to remove -- that 3B lost
two thirds of its score against an otherwise-worse 32k model, so the grade tracked the sizing
decision rather than the model.

```
GREEN   N_out >= 4,096    never cut off in practice
YELLOW  N_out >= 1,536    fine for shorter queries
RED     N_out >=   250    only a simple, direct answer
REFUSE  N_out <    250    cannot answer at all -- not installed
```

Context has not stopped mattering, it just enters in its proper place rather than dominating.
Grading on how large an *input* fits made the verdict mostly a restatement of the context size,
which is not the interesting question -- prefill is cheap next to generation. Now context bites
twice: it slows every generated token, and it caps the grade outright.

```
context > 256k   -> no cap, green reachable
context  16k-256k -> at most yellow
context   4k-16k  -> at most red
context <  4k     -> refused; the system prompt alone does not fit
```

That context is the **effective** one: the smaller of what the model was trained for and what
runtime sizing could actually afford on this device's memory. A context this device cannot serve is
not one the user gets, so on a 16GB board even a 1M-context model is graded on the ~75k it can
really hold -- which means green is genuinely out of reach on small hardware, by design.
`benchmark` re-runs sizing before measuring so that number never goes stale.

A turn's prefill is charged at ~2k tokens, not at the full context, and that distinction is
load-bearing. llama.cpp reuses the KV cache across turns, so a conversation grown to 131k prefills
only its new tokens. Charging the whole context instead cost 630s of prefill for a 131k model on a
real Jetson -- over budget before a single output token -- and rejected every large-context model
for being large-context, the exact inversion this scheme exists to remove.

Two details worth knowing, both of which came out of measuring real hardware rather than reasoning:

- **The depth fit holds up a long way out.** Checked against real `llama-bench` runs on a Jetson,
  a line fitted from depths 512 and 4096 predicted the measured cost at 16k within 1.1% and at 32k
  within 1.8% -- an 8x extrapolation. That is why the grading depth can sit at 32k without needing
  a 32k-deep probe every time.
- **The shallow measurement is taken at depth 512, not 0.** Decode cost per token nearly doubles
  over the first few hundred tokens and then flattens (17.3ms at depth 0 vs 32.8ms at 512 and
  33.9ms at 1024, measured on a CPU-only laptop), because generating at depth 0 does almost no
  attention work. Anchoring the fit there made it *optimistic* across the whole range the app
  actually uses -- the direction that promises turns which then time out.
- **Cold load counts.** `--models-max 1` makes every model switch a fresh load, and at pull time the
  file is still hot in the page cache, so a load timed right then measures RAM rather than disk.
  `posix_fadvise(DONTNEED)` drops it first and a real sequential read is timed instead.
- **The safety factor is measured, not guessed.** Checked against real full turns, the fit
  under-predicted by 1.26x at a 4,096-token input and 1.36x at 1,024 -- llama-bench
  excludes tokenization and sampling from its numbers, nothing measures the chat template on top,
  and real cost grows a little faster than linearly past the deepest point the probe can afford.
  At the original 1.25x that left a turn at `N_fit` really taking ~169s against a 155s budget, so
  the grade would have promised something false. It is 1.5x (1.75x for the live-server fallback),
  which puts the same turn at ~141s.

A refusal **deletes the model**, via the router's own `DELETE /models` (which reuses llama.cpp's
cache logic to clear the snapshot, its symlinks and the orphaned blobs). Merely withholding a preset
would not be enough: the router auto-discovers everything in the Hugging Face cache, so a refused
model would still be listed by `/v1/models` and selectable in the chat UI. `pull --force` keeps it
anyway, and `--skip-benchmark` skips the measurement entirely.

Failing to measure never becomes a refusal -- a busy machine, a missing binary or a failed probe
records no grade and keeps the model, because that says nothing about the model. Source-built
installs (Jetson-class arm64+CUDA, old glibc) historically had no `llama-bench` at all; they now
build it alongside `llama-server`, and any install still without it is measured through the running
router instead (`model_perf_live.py`), labelled as such and graded more conservatively.

`aipotluck-local-client benchmark [model]` re-measures without re-downloading -- the way back from
a skipped grade or one gone stale after a llama.cpp upgrade or a re-size. Grades live in
`llama_presets.performance.json` beside the presets file, with a device fingerprint so a stale one
is shown as stale rather than silently trusted.

**Validating the extrapolation is part of the work, not an afterthought:**
`scripts/validate_model_perf.py` predicts a grid of turn sizes, then actually runs turns at those
sizes and fails when the predictions do not hold -- in particular when they are optimistic, which
is the only direction that hurts users.

### Router mode and memory budgeting

`--models-max` (default `1`, `aipotluck.service.runner.DEFAULT_MODELS_MAX`) caps how many models
the router may hold loaded simultaneously. The sizing math above assumes *one* model gets the
whole memory budget -- raising `--models-max` is a real capacity tradeoff (multiple models sharing
one memory budget), not a free win, and isn't accounted for by anything in this repo today.

## HTTP API: remote model management

Everything `pull` / `list` / `benchmark` / `remove` do is also reachable over HTTP, on the same
local service port the status endpoints use (`127.0.0.1:8069` by default). Both surfaces call one
implementation (`aipotluck/service/model_ops.py`), so the preflight size check, the viability
screen, the LocalScore probe and the out-of-memory history all behave identically whichever one you
use -- there is no second code path to drift.

**Security: these endpoints are unauthenticated, exactly like `/status` and `/capabilities`.** The
service binds to localhost, and anything remote is expected to arrive through an authenticated
Pangolin resource (Pangolin SSO-gates public resources by default). That is the whole protection:
**mapping a public Pangolin resource straight at this port exposes model deletion to whoever can
reach it.** Nothing here checks a token.

| Method | Path | Returns | Notes |
|---|---|---|---|
| `GET` | `/models` | `200` listing | sizing, score, staleness and held-back state per model |
| `POST` | `/models` | `202` job, or `409` | body `{"model": "...", ...}` |
| `POST` | `/models/benchmark` | `202` job | body `{"model": "..."}`, or empty to measure everything |
| `DELETE` | `/models?model=<urlencoded>` | `200` | synchronous |
| `GET` | `/jobs` / `/jobs/{id}` | `200`, or `404` | newest first |

**Slow operations return a job; fast ones answer directly.** A pull downloads gigabytes, sizes,
screens and scores -- comfortably past ten minutes on a Jetson, which no client will hold a
connection open for through a tunnel. So `POST` returns `202` with a job id immediately and you poll
`/jobs/{id}`, where `state` moves `queued` → `running` → `succeeded`/`failed` and `progress` fills
in as the work happens. `list` and `remove` finish in milliseconds and answer inline.

**Jobs run one at a time, deliberately.** A benchmark taken while something else is loading a model
measures the contention rather than the model, and two models resident at once was an outright
allocation failure on a 16GB board. Parallel jobs would reintroduce both, and silently -- the
scores would simply be wrong, with nothing to say so.

```bash
# What is installed, and how it performs
curl -s localhost:8069/models | jq '.models[] | {id, ctx_size, score: .score.localscore, held_back}'

# Pull, then watch it
JOB=$(curl -s -XPOST localhost:8069/models \
        -H 'Content-Type: application/json' \
        -d '{"model":"bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"}' | jq -r .job.id)
curl -s localhost:8069/jobs/$JOB | jq '{state, progress: [.progress[].message], result}'

# Re-measure everything
curl -s -XPOST localhost:8069/models/benchmark -d '{}' -H 'Content-Type: application/json'

# Remove one (the id needs url-encoding -- it contains "/" and ":")
curl -s -XDELETE "localhost:8069/models?model=bartowski%2FQwen2.5-0.5B-Instruct-GGUF%3AQ4_K_M"
```

**A model too big for the machine is refused before anything is downloaded**, in the response you
are already waiting on rather than as a job that fails minutes later:

```json
{"error": {"code": "too_large_for_device",
           "message": "... is 39.6GB of weights, and this device has 31.0GB of memory in total ...",
           "preflight": {"fits": false, "model_bytes": 42520398432, "total_memory_bytes": 33272057856},
           "retry_with": {"allow_oversized": true}}}
```

`allow_oversized` and `keep_rejected` are separate on purpose, because they answer different
questions at different costs: one says "download it even though the weights do not fit", decided
before spending anything; the other says "keep it even though it cannot serve a turn here", decided
after measuring. `"force": true` sets both, matching the CLI's `--force`.

Errors are always `{"error": {"code", "message"}}`. `code` is stable and worth branching on:
`no_llama_install` (503), `bad_request` (400), `unknown_model` (404), `unknown_job` (404),
`too_large_for_device` (409), `download_failed` (502), `delete_failed` (503), `cache_unreadable`
(503). A failed job carries the same value in `error_code`.

## CLI on PATH

`aipotluck-local-client` (`aipotluck/installer/cli_shim.py`) is written
during install to `~/.local/bin/aipotluck-local-client` (`--system`:
`/usr/local/bin`) on Linux/macOS, or `%LOCALAPPDATA%\aipotluck\bin\aipotluck-local-client.cmd`
(`--system`: `%ProgramFiles%\aipotluck\bin`) on Windows -- a small wrapper
that execs the Python interpreter the installer resolved against
`cli.py`'s own file path directly, so it works regardless of what
`python3` means in whatever shell you're in, and regardless of CWD (no
`-m`/PYTHONPATH dependency). Not a `pip install`-based console script:
this project has no build step anywhere else either, and downloads
prebuilt binaries rather than building from source (same reasoning as
llama.cpp/newt).

On Linux/macOS, the per-user target (`~/.local/bin`) is only added to your
PATH automatically on the installer's best-effort check -- if it warns that
it isn't there, add it to your shell profile once:
```bash
export PATH="$HOME/.local/bin:$PATH"
```
On Windows the installer edits your user `PATH` in the registry directly
(no elevation needed) and broadcasts the change; a brand new terminal
always sees it even if the current one doesn't. `--system` on any OS
writes to a machine-wide location that's normally already on PATH
(`/usr/local/bin`) or prints the manual step to add it (Windows, since
that needs elevation this process may not have).

## Tests

```bash
pip install -e ".[dev]"    # or: pip install --user pytest
pytest
```

No real network and no real systemd/launchd. Every OS/network/service-manager boundary
(`fetch`/`newt_fetch`, `get_service_manager`, `LlamaSupervisor`/`NewtSupervisor`,
`urllib.request.urlopen`) is mocked or swapped for a lightweight fake; only real filesystem
writes happen, always rooted under pytest's own `tmp_path` -- the suite never touches the actual
per-OS install locations (`~/.local/bin`, `~/.config/aipotluck`, a real systemd unit, etc.). One
deliberate exception: `test_model_pull.py`, `test_model_sizing.py`, and `test_source_build.py` run
a real subprocess (a tiny stand-in script playing the part of `llama-server`/`cmake`) and, for the
last of those, a real spawned grandchild process to prove a build timeout actually kills the whole
process tree -- spawn/health-poll/terminate (or configure/build/kill), or in `model_sizing.py`'s
case parsing a real process's real captured startup log, is exactly the kind of thing a mock can
make *look* correct while testing nothing, so those modules are exercised for real instead. Covers
`platform_detect.py` (including glibc/CUDA-compute-capability detection), `build_strategy.py`'s
prebuilt-vs-source-build decision, `build_cache.py`'s custom-binary-cache lookup, `layout.py`,
`cli.py` (including the argparse regression -- see `test_cli_argparse.py`'s docstring),
`cli_shim.py`, `model_pull.py`'s `pull`/`list` orchestration, `model_sizing.py`'s probe/compute/
preset-writing pipeline, `model_presets.py`'s real INI/JSON read-modify-write (including that an
unrelated hand-written section survives a write), `install.py`'s `--no-service`/source-build/
binary-cache/real install paths (plus the default-model pre-fetch/pre-size step), `service/
runner.py`'s login-gating, router-mode arg-building, and startup preset backfill,
`/status` secret redaction and the real-HTTP `/capabilities` route, `diagnostics.py`'s per-section
failure isolation and its real-file per-model `runtime_params` view, `scripts/
package_custom_build.py`'s real relocatability check (a real fake `llama-server` script, copied to
a real different path and actually executed -- the same reasoning as the real-subprocess tests
above: this check exists specifically to catch a failure mode a static/mocked check would miss),
and `newt_supervisor.py`'s `tunnel_connected` (a real fake newt binary, a real health file, and a
real spawned subprocess proving `--health-file` is actually on the command line and stale state
is actually cleared -- not just that the code reads that way).

Not covered: the OS-native `ServiceManager` backends themselves (`systemd.py`/`launchd.py`/
`windows_service.py` — installing a real unit/plist/Scheduled Task), and the real download+
checksum+extract path in `fetch.py`/`newt_fetch.py`. Both need a real OS service manager or real
network access respectively to verify meaningfully; see this repo's own `ARCHITECTURE.md` for how
those were verified by hand instead (real installs, real `systemctl`/`launchctl`, a real model
download and inference request).

## Repo layout

```
vendor/llama.cpp/              git submodule, pinned commit (reference + docs)
llama_version.json             pinned release tag + per-platform asset checksums
llama_custom_builds.json        our own cached prebuilt binaries (see "Building from source")
newt_version.json               pinned newt (Pangolin tunnel client) release + per-platform checksums
install.sh                     public one-line installer entry point (Linux/macOS)
install.ps1                    public one-line installer entry point (Windows)
pyproject.toml                  packaging metadata -- console-script entries for a `pip install .` path only
aipotluck/                      the root package everything below lives under
  diagnostics.py                 GET /capabilities fingerprint -- reuses installer/ detection code
  installer/
    install.py                    installer CLI entry point -- always produces a logged-out install
    cli.py                         login/logout/status CLI for an already-installed device
    cli_shim.py                    writes the `aipotluck-local-client` wrapper onto PATH (see "CLI on PATH")
    platform_detect.py             OS/arch/GPU-backend/glibc/CUDA-compute-capability detection
    build_strategy.py              decide prebuilt-asset vs. build-from-source per host
    source_build.py                configure+build llama-server from vendor/llama.cpp
    build_cache.py                  check our own prebuilt-binary cache before compiling
    fetch.py                       download+checksum+extract llama.cpp releases
    newt_fetch.py                  download+checksum-verify the pinned newt binary
    model_pull.py                   `pull`/`list`: download a model ahead of time via -hf, list what's cached
    model_sizing.py                 auto-size ctx_size/parallel/cache_type_k/-v per model (router mode, CUR-1965)
    model_presets.py                read/write the --models-preset INI + its sibling tuning-reasons JSON
    layout.py                      per-OS install paths
    python_bootstrap.py            find/install a suitable Python (Windows: via winget)
    service/
      base.py                       ServiceManager ABC
      systemd.py                    Linux (tested)
      launchd.py                    macOS (stub, untested)
      windows_service.py            Windows: schtasks (no-admin) + pywin32 (--system)
  service/
    runner.py                       portable AipotluckServiceRunner (HTTP status + both supervisors,
                                     gated on runtime.json's "logged_in" flag)
    llama_supervisor.py             LlamaSupervisor: starts/restarts/health-polls llama-server
    newt_supervisor.py              NewtSupervisor: starts/restarts the newt tunnel client
    aipotluck_service.py            CLI entry point (systemd/launchd/schtasks invoke this directly)
    windows_service_host.py         pywin32 ServiceFramework wrapping the same runner
packaging/
  windows/install.ps1             Windows bootstrapper (installs Python if missing, then installs)
  windows|macos|linux/README.md   per-OS implementation notes
scripts/
  package_custom_build.py         standardized custom-binary-cache packaging tool (see "Building from source")
tests/                            unit tests -- see "Tests" above
ARCHITECTURE.md                  full design doc
```
