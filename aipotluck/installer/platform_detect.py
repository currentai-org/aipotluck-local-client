"""Platform, architecture, and GPU-backend detection.

Everything here is best-effort and non-fatal: if a probe fails or is
inconclusive, we fall back to a safe default (plain CPU asset for the
detected OS/arch). Detection never raises for a supported OS; unsupported
OSes raise UnsupportedPlatformError with a clear message.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from aipotluck.proc import NO_WINDOW

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class UnsupportedPlatformError(RuntimeError):
    pass


@dataclass(frozen=True)
class HostProfile:
    os_name: str      # "linux" | "macos" | "windows"
    arch: str         # "x64" | "arm64"
    backend: str      # "cpu" | "cuda" | "vulkan" | "rocm" | "metal"

    @property
    def asset_key(self) -> str:
        return f"{self.os_name}-{self.arch}-{self.backend}"


def detect_os() -> str:
    system = platform.system()
    if system == "Linux":
        return "linux"
    if system == "Darwin":
        return "macos"
    if system == "Windows":
        return "windows"
    raise UnsupportedPlatformError(f"Unsupported OS: {system!r}")


def running_under_rosetta() -> bool:
    """True when this process is an Intel (x86_64) binary being translated by Rosetta 2 on an Apple
    silicon Mac -- e.g. the GUI's Intel build opened on an M-series Mac, with its bundled Intel
    Python. `sysctl.proc_translated` is 1 then, 0 for a native process, and absent on Intel Macs
    and everywhere else."""
    if platform.system() != "Darwin":
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if libc.sysctlbyname(b"sysctl.proc_translated", ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
            return False
        return value.value == 1
    except (OSError, AttributeError):
        return False


def detect_arch() -> str:
    """The MACHINE's architecture, not this process's. Under Rosetta, platform.machine() says
    x86_64 on an Apple silicon Mac, which installed Intel llama.cpp and newt there: llama.cpp
    without Metal, translated instruction by instruction, so inference ran many times slower than
    the Mac can. The binaries this installs are separate processes, so they run natively whatever
    architecture the Python driving them is."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        if running_under_rosetta():
            return "arm64"
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    raise UnsupportedPlatformError(f"Unsupported architecture: {machine!r}")


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _cmd_ok(args: list[str]) -> bool:
    try:
        result = subprocess.run(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, **NO_WINDOW
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def detect_gpu_backend(os_name: str, arch: str, requested: str = "auto") -> str:
    """Return one of "cpu" | "cuda" | "vulkan" | "rocm" | "metal".

    requested: "auto" to probe, or an explicit override which is returned
    as-is (validated later against the asset manifest).
    """
    if requested != "auto":
        return requested

    if os_name == "macos":
        # Nothing to probe: upstream's one Apple-silicon release asset (llama-<tag>-bin-macos-arm64)
        # is built with Metal, so an arm64 Mac gets Metal from the same download. Named "metal" so
        # the asset key, `status` and the installer's summary say what the binary actually uses,
        # rather than "cpu" (which is what this returned before, and what misled a reader into
        # thinking a Mac ran inference on the CPU). Intel Macs stay "cpu": their asset is the
        # plain CPU build as far as this project has verified.
        return "metal" if arch == "arm64" else "cpu"

    if os_name == "windows":
        # CUDA only when the NVIDIA driver can run the CUDA version llama.cpp's Windows CUDA build
        # was compiled for (a driver that's too old fails at load, not at download); otherwise
        # Vulkan, which every current GPU driver ships, when there's a real GPU; otherwise CPU.
        if arch == "x64":
            need = windows_cuda_build_version()
            have = detect_driver_cuda_version()
            if need and have and _version_tuple(have) >= _version_tuple(need):
                return "cuda"
            if detect_windows_vulkan_gpu():
                return "vulkan"
        return "cpu"

    if os_name == "linux":
        if _has_cmd("nvidia-smi") and _cmd_ok(["nvidia-smi"]):
            return "cuda"
        if _has_cmd("rocminfo") and _cmd_ok(["rocminfo"]):
            return "rocm"
        # Vulkan is a reasonable generic GPU fallback but a working Vulkan
        # loader doesn't guarantee a usable inference backend, and picking
        # it wrong is worse than CPU (which always works). Only opt in to
        # vulkan when explicitly requested by the caller.
        return "cpu"

    return "cpu"


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:2])


def windows_cuda_build_version() -> str | None:
    """The CUDA version the pinned windows-x64-cuda asset was built with ("12.4"), from
    llama_version.json -- or None if that asset isn't pinned."""
    try:
        manifest = json.loads((_REPO_ROOT / "llama_version.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = manifest.get("assets", {}).get("windows-x64-cuda") or {}
    value = entry.get("cuda_version")
    return value if isinstance(value, str) else None


def detect_driver_cuda_version() -> str | None:
    """The newest CUDA version the installed NVIDIA driver supports ("12.8"), as nvidia-smi's
    banner reports it, or None with no NVIDIA driver (or an unreadable one). A CUDA build runs on
    any driver whose number here is >= the version it was compiled with."""
    if not _has_cmd("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
            text=True, **NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
    return m.group(1) if m else None


# Adapters Windows lists that are not a GPU anyone could run inference on.
_NOT_A_GPU = ("microsoft basic display", "microsoft basic render", "microsoft remote display",
              "microsoft hyper-v video", "citrix", "parsec virtual", "virtual display")


_DISPLAY_CLASS_KEY = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def windows_display_adapters() -> list[str]:
    """Names (DriverDesc) of the display adapters with a driver installed, from the display device
    class in the registry. Readable by a standard user in any session -- unlike
    EnumDisplayDevicesW, which sees nothing outside an interactive desktop (an SSH logon, a
    service), and WMI, which a restricted logon can't use. [] off Windows or if it can't be read."""
    try:
        import winreg
    except ImportError:
        return []
    names: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS_KEY) as cls:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(cls, i)
                except OSError:
                    break
                if not sub.isdigit():  # "Properties" and similar aren't devices
                    continue
                try:
                    with winreg.OpenKey(cls, sub) as dev:
                        desc, _ = winreg.QueryValueEx(dev, "DriverDesc")
                except OSError:
                    continue
                if isinstance(desc, str) and desc and desc not in names:
                    names.append(desc)
    except OSError:
        return []
    return names


def detect_windows_vulkan_gpu() -> bool:
    """True when this Windows machine has a real GPU and a Vulkan loader (vulkan-1.dll, which
    NVIDIA, AMD and Intel drivers all install). Doesn't prove the driver's Vulkan works -- nothing
    short of loading a model does -- but rules out the common no-GPU cases: VMs, remote sessions
    and servers on Microsoft's basic display driver."""
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    if not (system32 / "vulkan-1.dll").exists():
        return False
    return any(not n.lower().startswith(_NOT_A_GPU) for n in windows_display_adapters())


def detect_host_profile(requested_backend: str = "auto") -> HostProfile:
    os_name = detect_os()
    arch = detect_arch()
    backend = detect_gpu_backend(os_name, arch, requested_backend)
    return HostProfile(os_name=os_name, arch=arch, backend=backend)


def detect_glibc_version() -> tuple[int, int] | None:
    """(major, minor) of the host's glibc, or None -- not glibc (musl, macOS, Windows) or the
    version string didn't parse. Never raises; callers must treat None as "unknown, don't block
    on it" rather than "definitely incompatible"."""
    lib, ver = platform.libc_ver()
    if lib != "glibc" or not ver:
        return None
    parts = ver.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def glibc_satisfies(min_glibc: str | None, host_glibc: tuple[int, int] | None) -> bool:
    """True unless we can PROVE the host's glibc is older than min_glibc. Either side being
    unknown (no floor recorded, or host isn't glibc-based / couldn't be read) means "don't block" --
    this check exists to catch a real, confirmed failure mode (llama.cpp's published arm64 Linux
    release asset needs glibc 2.38; JetPack 6.2.3 / Ubuntu 22.04 ships 2.35 and the binary won't
    even start), not to invent a new one out of a probe we can't complete."""
    if min_glibc is None or host_glibc is None:
        return True
    major_str, minor_str = min_glibc.split(".")
    return host_glibc >= (int(major_str), int(minor_str))


def detect_cuda_compute_capability() -> str | None:
    """The running NVIDIA GPU's compute capability as "M.m" (e.g. "8.7" on Jetson Orin, "8.9" on
    an RTX 4090), or None if nvidia-smi is absent/fails/unparseable. This is deliberately a
    separate, more detailed probe than detect_gpu_backend's plain "is cuda usable at all" check --
    it's only needed when a source build must pick a CMAKE_CUDA_ARCHITECTURES value, which a
    prebuilt-asset install never has to do."""
    if not _has_cmd("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5, text=True, **NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    value = lines[0].split(",")[0].strip()
    if "." not in value:
        return None
    return value
