"""Build llama-server from source when build_strategy decides no prebuilt asset is viable.

Confirmed viable against a real Jetson Orin Nano (JetPack 6.2.3): a manually-built llama-server
there already links libggml-cuda.so against the JetPack-bundled CUDA 12.6 toolkit at
CMAKE_CUDA_ARCHITECTURES=87 and serves real inference. This module automates exactly that path:
check the host actually has what a build needs (never silently `sudo apt-get install` on the
caller's behalf -- many hosts, including that Jetson, require an interactive sudo password, so a
non-interactive installer would just hang or fail confusingly), check out the pinned tag into the
vendor/llama.cpp submodule, configure + build with resource-aware parallelism, and locate the
resulting binary the same way fetch.py locates one inside an extracted release archive.

The full set of packages this checks for (see missing_apt_packages) was cross-referenced against
llama.cpp's own release CI, not guessed from one host's missing package -- every Linux
build-*.yml/release.yml job (cpu, cuda, vulkan, rocm) was read for its own `apt-get install` line,
intersected with what a native GGML_NATIVE=ON build of just the llama-server target actually needs
(their CI installs several things -- ninja-build, python3-venv, git-lfs, libjpeg-dev -- that are
there for CI's own portable multi-arch/test/conversion-script concerns, none of which apply here;
confirmed libjpeg-dev in particular isn't a real llama.cpp build dependency at all, by checking that
nothing in the CMake tree does find_package(JPEG) -- image loading goes through the header-only
stb_image instead).

Two build-time gotchas are worth a full explanation, both the same shape (a plain, non-REQUIRED
find_package(...) that degrades SILENTLY -- configure still succeeds -- rather than failing loud):

- llama.cpp's HTTPS support (`-hf`/`--cache-list`, which this project's own pull/list commands
  depend on) needs libssl-dev; without it, configure succeeds and the gap only shows up later as a
  runtime error the first time something tries to download a model.
- OpenMP multi-threading on the CPU backend (ggml/src/CMakeLists.txt's `find_package(OpenMP)`)
  needs libgomp1; without it, configure succeeds with only a `message(WARNING "OpenMP not found")`
  buried in build output, and the binary silently falls back to single-threaded CPU inference --
  no error, just much slower than it should be.

Preflight checks for both before ever starting cmake, specifically to avoid either burning a
30-90 minute build on a binary with a silent gap, or only noticing the slowdown/breakage after.
"""

from __future__ import annotations

import ctypes.util
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from aipotluck.installer.fetch import find_binary

log = logging.getLogger("aipotluck.installer.source_build")

DEFAULT_BUILD_TIMEOUT_SECONDS = 5400  # 90 min -- a cold CUDA build on an 8GB Jetson-class board is slow.
MIN_FREE_DISK_GB = 6
_OPENSSL_HEADER_CANDIDATES = (
    Path("/usr/include/openssl/ssl.h"),
    Path("/usr/local/include/openssl/ssl.h"),
)
_VULKAN_HEADER_CANDIDATES = (
    Path("/usr/include/vulkan/vulkan.h"),
    Path("/usr/local/include/vulkan/vulkan.h"),
)


class BuildError(RuntimeError):
    pass


def _find_cxx_compiler() -> str | None:
    return shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")


def find_nvcc() -> Path | None:
    """nvcc isn't reliably on PATH even when the CUDA toolkit is genuinely installed -- confirmed
    live on the reference Jetson: `which nvcc` came back empty while
    /usr/local/cuda-12.6/bin/nvcc worked fine and reported a real version. Prefer PATH when it's
    there, then fall back to the standard /usr/local/cuda*/bin layout (checked newest-first, since
    a host can have several toolkit versions installed side by side)."""
    on_path = shutil.which("nvcc")
    if on_path:
        return Path(on_path)
    for candidate in sorted(Path("/usr/local").glob("cuda*/bin/nvcc"), reverse=True):
        if candidate.is_file():
            return candidate
    return None


def _has_openssl_headers() -> bool:
    if any(p.exists() for p in _OPENSSL_HEADER_CANDIDATES):
        return True
    if shutil.which("pkg-config"):
        try:
            return subprocess.run(
                ["pkg-config", "--exists", "openssl"], timeout=5,
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
    return False


def _has_libgomp() -> bool:
    return ctypes.util.find_library("gomp") is not None


def find_glslc() -> str | None:
    return shutil.which("glslc")


def _has_vulkan_headers() -> bool:
    return any(p.exists() for p in _VULKAN_HEADER_CANDIDATES)


def find_hipcc() -> str | None:
    return shutil.which("hipcc")


_APT_PACKAGE_PROBLEMS = {
    "cmake": "cmake not found",
    "git": "git not found",
    "build-essential": "no C++ compiler found",
    "libssl-dev": (
        "OpenSSL development headers not found -- llama-server's -hf downloader and "
        "--cache-list (this CLI's pull/list commands) compile out silently without them, with "
        "no build error, only a later runtime one"
    ),
    "libgomp1": (
        "OpenMP runtime (libgomp) not found -- ggml's find_package(OpenMP) degrades silently the "
        "same way OpenSSL does: configure succeeds with only a build-log warning, and the CPU "
        "backend falls back to single-threaded inference with no error, just much slower than it "
        "should be"
    ),
}


def missing_apt_packages() -> list[str]:
    """The subset of build gaps a single `apt-get install` can actually fix -- cmake, git, a C++
    compiler, OpenSSL headers, the OpenMP runtime. Cross-referenced against every Linux
    build-*.yml/release.yml job in llama.cpp's own CI (cpu/cuda/vulkan/rocm all install this same
    core set); see this module's docstring for what CI installs beyond this that isn't actually
    needed here. Deliberately excludes any GPU vendor toolchain (CUDA's nvcc, Vulkan's SDK
    components, ROCm's hipcc/rocblas/etc.): none of those are a small, safe auto-install -- they're
    multi-GB, vendor-repo-only, and on Jetson/JetPack the CUDA toolkit specifically is the
    vendor-managed OS image. Those gaps always stay an instruction (see check_build_prerequisites),
    never something install_apt_packages installs, no matter how apt-install is invoked."""
    packages = []
    if shutil.which("cmake") is None:
        packages.append("cmake")
    if shutil.which("git") is None:
        packages.append("git")
    if _find_cxx_compiler() is None:
        packages.append("build-essential")
    if not _has_openssl_headers():
        packages.append("libssl-dev")
    if not _has_libgomp():
        packages.append("libgomp1")
    return packages


def has_apt() -> bool:
    return shutil.which("apt-get") is not None


class AptInstallError(BuildError):
    pass


def install_apt_packages(packages: list[str], *, timeout: float = 300) -> None:
    """Install `packages` via `sudo apt-get update && sudo apt-get install -y <packages>`, with
    stdin/stdout/stderr all left inherited from this process (never captured or redirected) --
    `sudo` itself talks to the controlling terminal directly to prompt for a password, and apt's
    own progress should be visible the same way a long build's is. Only ever called after the
    caller has gotten explicit consent (install.py's --allow-apt-install flag or an interactive
    prompt) -- this function itself does not ask, so it must never be wired to run unconditionally.
    """
    if not packages:
        return
    log.info("Installing missing build dependencies via apt: %s", " ".join(packages))
    try:
        update = subprocess.run(["sudo", "apt-get", "update"], timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AptInstallError(f"'sudo apt-get update' did not run: {exc}") from exc
    if update.returncode != 0:
        raise AptInstallError(f"'sudo apt-get update' failed (exit {update.returncode})")
    try:
        install = subprocess.run(["sudo", "apt-get", "install", "-y", *packages], timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AptInstallError(f"'sudo apt-get install' did not run: {exc}") from exc
    if install.returncode != 0:
        raise AptInstallError(
            f"'sudo apt-get install -y {' '.join(packages)}' failed (exit {install.returncode})"
        )


def check_build_prerequisites(*, backend: str) -> list[str]:
    """Return a list of human-readable problems; empty means the host is ready to build. Every
    entry names both the gap and the exact fix (never just "cmake missing") -- for the
    apt-fixable ones (see missing_apt_packages), the caller may offer to fix them automatically
    with explicit consent; a GPU vendor toolchain gap (nvcc/Vulkan SDK components/hipcc) never
    gets that treatment, see missing_apt_packages' docstring for why."""
    problems = [
        f"{_APT_PACKAGE_PROBLEMS[pkg]} (sudo apt-get install -y {pkg})"
        for pkg in missing_apt_packages()
    ]
    if backend == "cuda" and find_nvcc() is None:
        problems.append(
            "nvcc not found -- CUDA was detected (nvidia-smi works) but the CUDA toolkit compiler "
            "isn't on PATH or under /usr/local/cuda*/bin. On JetPack this ships with the OS image; "
            "check `ls /usr/local/cuda*/bin/nvcc` and add its directory to PATH."
        )
    if backend == "vulkan" and (find_glslc() is None or not _has_vulkan_headers()):
        problems.append(
            "Vulkan SDK components not found (glslc shader compiler and/or vulkan/vulkan.h) -- "
            "these come from distro/vendor-specific packages (e.g. Debian/Ubuntu: libvulkan-dev "
            "plus glslc from the LunarG Vulkan SDK, package names differ by release) rather than "
            "one generic apt package this installer can safely guess. Install them for your "
            "distro, then re-run."
        )
    if backend == "rocm" and find_hipcc() is None:
        problems.append(
            "hipcc not found -- ROCm was requested but its dev toolchain isn't on PATH. Like the "
            "CUDA toolkit, ROCm comes from AMD's own repo/installer, not a plain apt package this "
            "installer can safely install automatically."
        )
    return problems


def check_free_disk_gb(path: Path, *, minimum_gb: float = MIN_FREE_DISK_GB) -> float | None:
    """Free space (in GiB) at `path`, or None if it can't be determined. Doesn't raise on its
    own -- callers decide what "not enough" means; this only exists because a from-source build
    was confirmed to run on a Jetson with only 14GB free system-wide, and a build that dies
    two-thirds through from ENOSPC is a much worse experience than refusing up front."""
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except OSError:
        return None


def recommended_jobs(*, want_cuda: bool) -> int:
    """A conservative -j value. nvcc is memory-hungry enough that the naive `-j$(nproc)` default
    can OOM (or thrash into swap, which is slower than fewer parallel jobs) on an 8GB board like
    the Jetson Orin Nano this was built against -- 6 cores but only ~5.6GB available in practice.
    Budgets more RAM per job for a CUDA build (nvcc) than a plain C++ compile."""
    cpu_count = os.cpu_count() or 1
    available_gb = _available_memory_gb()
    if available_gb is None:
        return cpu_count
    budget_per_job = 2.0 if want_cuda else 1.0
    return max(1, min(cpu_count, int(available_gb // budget_per_job)))


def _available_memory_gb() -> float | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _total_memory_gb() -> float | None:
    """Installed physical RAM -- a stable hardware fact, unlike _available_memory_gb() above,
    which fluctuates with whatever else is running. aipotluck.installer.model_sizing budgets
    against this deliberately, not a live "available now" snapshot -- see that module's own
    docstring for why."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _init_vendor_submodule(repo_root: Path, vendor_dir: Path) -> None:
    """Fetch llama.cpp's source on demand, the first time a build actually needs it.

    install.sh/install.ps1 deliberately clone WITHOUT --recurse-submodules. Most installs resolve
    to a prebuilt binary -- an upstream release asset or one of our own cached custom builds -- and
    never compile anything, so paying ~174MB of llama.cpp source up front is a download the great
    majority of users have no use for. This function is the one place that genuinely needs it, so
    this is where it gets paid.

    Shallow on purpose: ensure_source_checked_out immediately fetches and detaches onto the pinned
    tag anyway, so the submodule's own history is never read.
    """
    if not (repo_root / ".git").exists():
        raise BuildError(
            f"A source build needs llama.cpp's source at {vendor_dir}, but {repo_root} is not a git "
            "checkout so it cannot be fetched automatically. Re-run the installer (which clones "
            "this repo), or clone it yourself and retry."
        )
    log.info("Fetching llama.cpp source -- needed for a from-source build, downloaded only now")
    _run(
        ["git", "-C", str(repo_root), "submodule", "update", "--init", "--depth", "1",
         "--", "vendor/llama.cpp"],
        timeout=1800,
    )
    if not (vendor_dir / ".git").exists():
        raise BuildError(
            f"Initialising the llama.cpp submodule did not produce a checkout at {vendor_dir}. Run "
            f"'git -C {repo_root} submodule update --init --recursive' by hand to see why."
        )


def ensure_source_checked_out(repo_root: Path, tag: str) -> Path:
    """Point vendor/llama.cpp's submodule checkout at the pinned release tag, shallowly, fetching
    the submodule itself first if this install has never needed it before (see
    _init_vendor_submodule for why it is not fetched at clone time)."""
    vendor_dir = repo_root / "vendor" / "llama.cpp"
    if not (vendor_dir / ".git").exists():
        _init_vendor_submodule(repo_root, vendor_dir)
    log.info("Checking out llama.cpp %s into %s for source build", tag, vendor_dir)
    _run(["git", "-C", str(vendor_dir), "fetch", "--depth", "1", "origin", f"refs/tags/{tag}"], timeout=120)
    _run(["git", "-C", str(vendor_dir), "checkout", "--detach", "FETCH_HEAD"], timeout=30)
    return vendor_dir


def _git_commit(vendor_dir: Path) -> str:
    result = _run(["git", "-C", str(vendor_dir), "rev-parse", "HEAD"], timeout=15)
    return result.stdout.strip()


def _run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except OSError as exc:
        raise BuildError(f"Could not run {' '.join(args)}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"{' '.join(args)} did not finish within {timeout:.0f}s") from exc
    if result.returncode != 0:
        raise BuildError(f"{' '.join(args)} failed (exit {result.returncode}):\n{result.stderr.strip()}")
    return result


def _configure(source_dir: Path, build_dir: Path, *, cuda_arch: str | None, cmake_binary: str) -> None:
    args = [
        cmake_binary, "-S", str(source_dir), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        # LLAMA_BUILD_TOOLS must stay ON: the llama-server target itself lives under tools/server,
        # and the top-level CMakeLists only ever does add_subdirectory(tools) -- which is what
        # actually defines the "llama-server" target `cmake --build --target llama-server` asks
        # for -- when LLAMA_BUILD_TOOLS is on. LLAMA_BUILD_SERVER alone (tools/CMakeLists.txt's own
        # gate one level down) is never even evaluated otherwise. Confirmed live on the reference
        # Jetson: turning LLAMA_BUILD_TOOLS off (to shrink the build, on the wrong assumption that
        # it was independent of the server) configured fine but failed the actual build with
        # "No rule to make target 'llama-server'" -- the target had never been defined at all.
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_OPENSSL=ON",
        # Relocatable $ORIGIN rpath, matching llama.cpp's own release CI exactly (release.yml's
        # ubuntu-cuda job). Without this, cmake's default rpath points at this exact build_dir's
        # absolute path -- the binary works fine built-and-run in place (which is all a plain
        # from-source install ever does), but silently breaks the moment bin/ is archived and
        # extracted somewhere else, e.g. into a *different* host's install root, or packaged as a
        # custom binary-cache asset (build_cache.py) for reuse on another machine entirely.
        "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
    ]
    if cuda_arch:
        args += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"]
        # cmake's own CUDA-language detection only searches PATH -- confirmed live on the
        # reference Jetson: nvcc is genuinely installed but not on PATH there (it's under
        # /usr/local/cuda-12.6/bin), find_nvcc()'s glob fallback locates it fine, but without this
        # flag cmake still fails with CMAKE_CUDA_COMPILER-NOTFOUND. Always pass it when we have it
        # rather than only as a fallback -- it's strictly more robust than letting cmake redo a
        # weaker version of the same search.
        nvcc = find_nvcc()
        if nvcc:
            args += [f"-DCMAKE_CUDA_COMPILER={nvcc}"]
    log.info("Configuring: %s", " ".join(args))
    _run(args, timeout=300)


def _build(build_dir: Path, *, jobs: int, timeout: float, cmake_binary: str) -> None:
    _build_target(build_dir, "llama-server", jobs=jobs, timeout=timeout, cmake_binary=cmake_binary)

    # llama-bench is what model_perf grades a model with, and a source-built install is exactly the
    # slow arm64/old-glibc hardware where an unusable model is most likely -- so it is worth having
    # here rather than only in the prebuilt archives. It must be built BEFORE
    # _cleanup_intermediate_objects() rmtree's the CMakeFiles dirs, which is why it lives in this
    # same call rather than being added lazily later: afterwards it would cost a full rebuild.
    #
    # Best-effort on purpose. The heavy ggml/llama libraries are already compiled by this point, so
    # this is one extra translation unit and a link, but if some toolchain refuses it the install
    # must still succeed -- model_perf falls back to measuring through the running server, and a
    # missing benchmark binary is a degraded grade rather than a broken device.
    try:
        _build_target(build_dir, "llama-bench", jobs=jobs, timeout=timeout, cmake_binary=cmake_binary)
    except BuildError as exc:
        log.warning(
            "Could not build llama-bench (%s) -- the install is fine; model speed grading will "
            "measure through the running server instead.", exc,
        )


def _build_target(build_dir: Path, target: str, *, jobs: int, timeout: float, cmake_binary: str) -> None:
    args = [cmake_binary, "--build", str(build_dir), "--target", target, "-j", str(jobs)]
    log.info("Building %s with -j%d (this can take a long time -- output follows live)", target, jobs)
    # A build process tree (cmake -> make/ninja -> cc1/nvcc/ld) can be deep; start a new session so
    # a timeout can kill the *whole tree* via its process group, not just cmake's direct child,
    # which would otherwise leave orphaned compiler workers running. Output is left uncaptured
    # (inherits this process's stdout/stderr) for the same reason model_pull.py leaves llama-server's
    # own download progress uncaptured: a silent 30-90 minute wait reads as a hang.
    proc = subprocess.Popen(args, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        raise BuildError(f"Building {target} did not finish within {timeout:.0f}s (killed)")
    if proc.returncode != 0:
        raise BuildError(f"Building {target} failed (exit {proc.returncode}) -- see output above")


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        if proc.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    proc.wait(timeout=10)


def _cleanup_intermediate_objects(build_dir: Path) -> None:
    """Reclaim disk by deleting CMake's per-target object directories once the binary we actually
    want is built and located -- real motivation, not theoretical: the reference Jetson build this
    was validated against runs with only 14GB free system-wide, and a from-source CUDA build's
    object files are the single biggest thing this tool ever writes to disk (the hand-built
    reference on that device: 659MB). This forfeits `cmake --build`'s own incremental-rebuild
    cache; that's an acceptable trade because our own idempotency (the fingerprint marker) already
    treats "flags changed" as "rebuild from scratch," never as "relink.\""""
    for cmake_files_dir in build_dir.rglob("CMakeFiles"):
        if cmake_files_dir.is_dir():
            shutil.rmtree(cmake_files_dir, ignore_errors=True)


def build_llama_server(
    source_dir: Path,
    build_dir: Path,
    *,
    fingerprint: str,
    cuda_arch: str | None = None,
    jobs: int | None = None,
    timeout: float = DEFAULT_BUILD_TIMEOUT_SECONDS,
    cmake_binary: str = "cmake",
) -> Path:
    """Configure + build llama-server from `source_dir` into `build_dir`, returning the built
    binary's path. Idempotent across calls with the same `fingerprint` (e.g. "<git-sha>:<cuda_arch>")
    -- a matching marker plus an existing binary skips straight to returning it, so re-running the
    installer doesn't repeat a 90-minute build for no reason."""
    marker = build_dir / ".build_fingerprint"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == fingerprint:
        try:
            return find_binary(build_dir, "llama-server")
        except FileNotFoundError:
            log.warning("Build marker present but binary missing under %s; rebuilding", build_dir)

    jobs = jobs if jobs is not None else recommended_jobs(want_cuda=cuda_arch is not None)
    build_dir.mkdir(parents=True, exist_ok=True)
    _configure(source_dir, build_dir, cuda_arch=cuda_arch, cmake_binary=cmake_binary)
    _build(build_dir, jobs=jobs, timeout=timeout, cmake_binary=cmake_binary)

    binary = find_binary(build_dir, "llama-server")
    marker.write_text(fingerprint, encoding="utf-8")
    _cleanup_intermediate_objects(build_dir)
    log.info("Built llama-server: %s", binary)
    return binary


def ensure_llama_server_built(
    repo_root: Path,
    build_root: Path,
    tag: str,
    *,
    cuda_arch: str | None = None,
    jobs: int | None = None,
    timeout: float = DEFAULT_BUILD_TIMEOUT_SECONDS,
) -> Path:
    """Top-level entry point install.py calls: check out the pinned tag, then build. build_root is
    a persistent directory (e.g. under the install layout's install_root) distinct from the
    vendor/llama.cpp checkout itself, so multiple installs/backends don't clobber one build."""
    source_dir = ensure_source_checked_out(repo_root, tag)
    commit = _git_commit(source_dir)
    fingerprint = f"{commit}:{cuda_arch or 'cpu'}"
    return build_llama_server(
        source_dir, build_root, fingerprint=fingerprint, cuda_arch=cuda_arch, jobs=jobs, timeout=timeout,
    )
