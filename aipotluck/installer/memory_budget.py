"""Which memory a model's weights and KV cache will actually live in, and how much of it is free.

Budgeting against 80% of TOTAL system RAM regardless of backend is wrong on a discrete GPU, where it
is the wrong pool entirely: a 16GB Windows box with a 6GB GTX 1660 Ti sized Qwen3.5-4B to its full
262,144-token context, llama.cpp then pushed most layers off the GPU to make room for the KV cache,
and generation ran at ~4 tok/s instead of the ~40 the card manages with a context that fits.

So the budget comes from the pool the backend really allocates from:

- **vram** -- a discrete GPU (CUDA/Vulkan/ROCm on a card with its own memory): the device's free
  memory, as llama.cpp itself reports it (`llama-server --list-devices`).
- **unified** -- a GPU that shares system memory (Apple silicon/Metal, Jetson and other Tegra
  boards, DGX Spark, integrated GPUs): free system memory, since weights, KV cache and everything
  else on the machine come out of the same pool. This is also what llama.cpp's own CUDA backend
  reports for an integrated device (MemAvailable, see ggml-cuda.cu's UMA branch).
- **system** -- CPU-only builds, or a GPU we could not read: free system memory.

Free, not total: total RAM looks like "a stable hardware fact", but on a GPU
the free figure is what decides whether the layers land on it -- the desktop compositor alone held
~1GB of that 6GB card. A safety margin (`margin_bytes`) is kept back on top, matching llama.cpp's
own `--fit` default of 1GiB per device, or 10% of the pool if that is larger.

Never raises: anything that cannot be measured comes back as a budget of None with a reason, and
model_sizing falls back to its total-RAM estimate.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from aipotluck.proc import NO_WINDOW
from aipotluck.installer import source_build

log = logging.getLogger("aipotluck.installer.memory_budget")

GiB = 1024**3
MiB = 1024**2

KIND_VRAM = "vram"
KIND_UNIFIED = "unified"
KIND_SYSTEM = "system"

# llama.cpp's own --fit keeps 1GiB free per device by default (common.h: fit_params_target); 10% of
# the pool when that is larger, so a 64GB unified machine isn't sized to its last gigabyte.
_MARGIN_MIN_BYTES = 1 * GiB
_MARGIN_FRACTION = 0.10

# An accelerator reporting at least this share of system RAM as its own is sharing it: an
# integrated GPU's Vulkan heaps include system memory, and llama.cpp reports MemAvailable as a UMA
# CUDA device's memory. No discrete consumer card has 90% of its host's RAM.
_UNIFIED_SHARE = 0.90

_DEVICE_LINE = re.compile(r"^\s*([A-Za-z]+\d+):\s*(.*?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)\s*$")


@dataclass(frozen=True)
class Device:
    name: str          # llama.cpp's device name: "CUDA0", "Vulkan0", "ROCm0", "MTL0"
    description: str   # "NVIDIA GeForce GTX 1660 Ti"
    total_bytes: int
    free_bytes: int

    @property
    def backend(self) -> str:
        return re.sub(r"\d+$", "", self.name).lower()


@dataclass(frozen=True)
class MemoryBudget:
    kind: str                  # KIND_VRAM | KIND_UNIFIED | KIND_SYSTEM
    pool: str                  # what was measured, in words: "CUDA0 (NVIDIA GeForce GTX 1660 Ti)"
    free_bytes: int | None
    total_bytes: int | None
    reason: str                # why this pool -- a plain sentence for the tuning record

    @property
    def margin_bytes(self) -> int:
        return max(_MARGIN_MIN_BYTES, int((self.total_bytes or 0) * _MARGIN_FRACTION))

    @property
    def budget_bytes(self) -> int | None:
        """What a model's weights + KV cache may use: free memory less the safety margin."""
        if self.free_bytes is None:
            return None
        return self.free_bytes - self.margin_bytes

    def describe(self) -> str:
        if self.free_bytes is None:
            return f"{self.pool}: free memory unknown"
        return (
            f"{self.pool}: {self.free_bytes / GiB:.2f}GB free of {(self.total_bytes or 0) / GiB:.2f}GB, "
            f"{self.margin_bytes / GiB:.2f}GB kept back"
        )


# --------------------------------------------------------------------------- system memory


def system_memory_bytes() -> tuple[int | None, int | None]:
    """(available, total) physical memory in bytes. Available is what the OS could hand out now
    without swapping: MemAvailable on Linux, ullAvailPhys on Windows, free+inactive+speculative
    pages on macOS. Either can be None if it cannot be read."""
    total_gb = source_build._total_memory_gb()
    total = int(total_gb * GiB) if total_gb else None
    if Path("/proc/meminfo").exists():
        avail_gb = source_build._available_memory_gb()
        return (int(avail_gb * GiB) if avail_gb else None), total
    if os.name == "nt":
        return _windows_available_bytes(), total
    return _macos_available_bytes(), total


def _windows_available_bytes() -> int | None:
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
            (n, ctypes.c_uint64) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                          "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                          "ullAvailExtendedVirtual")
        ]

    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys) or None
    except (AttributeError, OSError, ValueError):
        return None


def _macos_available_bytes() -> int | None:
    vm_stat = shutil.which("vm_stat") or ("/usr/bin/vm_stat" if Path("/usr/bin/vm_stat").exists() else None)
    if vm_stat is None:
        return None
    try:
        out = subprocess.run([vm_stat], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page = re.search(r"page size of (\d+) bytes", out)
    if not page:
        return None
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            pages += int(m.group(1))
    return pages * int(page.group(1)) or None


# --------------------------------------------------------------------------- accelerators


def list_devices(server_binary: Path, timeout: float = 30.0) -> list[Device] | None:
    """The accelerators this llama.cpp build can use, as it reports them (`--list-devices`), or
    None if the binary could not be asked. An empty list means a CPU-only build or no usable GPU."""
    try:
        out = subprocess.run(
            [str(server_binary), "--list-devices"], capture_output=True, text=True,
            errors="replace", timeout=timeout, **NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Could not list llama.cpp's devices (%s)", exc)
        return None
    devices = []
    for line in (out.stdout + "\n" + out.stderr).splitlines():
        m = _DEVICE_LINE.match(line)
        if m:
            name, desc, total, free = m.groups()
            devices.append(Device(name, desc, int(total) * MiB, int(free) * MiB))
    if not devices and "Available devices" not in out.stdout:
        log.warning("llama-server --list-devices printed nothing recognisable (exit %s)", out.returncode)
        return None
    return devices


def _is_tegra() -> bool:
    return Path("/etc/nv_tegra_release").exists() or Path("/sys/module/tegra_fuse").exists()


def unified_reason(device: Device, system_total: int | None, os_name: str, arch: str) -> str | None:
    """Why this accelerator shares system memory, or None if it has its own."""
    if device.backend in ("mtl", "metal"):
        return "Apple silicon's GPU shares the system's unified memory"
    if device.backend == "cuda" and os_name == "linux" and (arch == "arm64" or _is_tegra()):
        return "an NVIDIA Tegra/Grace-class GPU (Jetson, DGX Spark) shares system memory"
    if system_total and device.total_bytes >= _UNIFIED_SHARE * system_total:
        return (
            f"{device.name} reports {device.total_bytes / GiB:.1f}GB, at least "
            f"{int(_UNIFIED_SHARE * 100)}% of the system's {system_total / GiB:.1f}GB -- an "
            "integrated GPU sharing system memory"
        )
    return None


def detect(server_binary: Path, os_name: str, arch: str) -> MemoryBudget:
    """The memory pool this llama.cpp build will put a model in, and how much of it is free."""
    available, total = system_memory_bytes()

    def system(reason: str) -> MemoryBudget:
        return MemoryBudget(KIND_SYSTEM, "system memory", available, total, reason)

    devices = list_devices(server_binary)
    if devices is None:
        return system("couldn't ask llama.cpp which devices it can use, so sized against free system memory")
    if not devices:
        return system("CPU-only llama.cpp build (no GPU device), so sized against free system memory")

    # llama.cpp splits across every device it finds; one GPU is what this project supports today,
    # so the first one (llama.cpp's own main device) decides.
    dev = devices[0]
    pool = f"{dev.name} ({dev.description})"
    why_unified = unified_reason(dev, total, os_name, arch)
    if why_unified:
        free = available if available is not None else dev.free_bytes
        return MemoryBudget(
            KIND_UNIFIED, f"{pool}, unified with system memory", free, total,
            f"{why_unified}, so sized against free shared memory",
        )
    return MemoryBudget(
        KIND_VRAM, pool, dev.free_bytes, dev.total_bytes,
        f"{dev.name} is a discrete GPU with its own memory, so sized against its free VRAM",
    )


def system_fallback(reason: str) -> MemoryBudget:
    """Free system memory, for a model whose weights don't fit the GPU at all (llama.cpp will
    split its layers between GPU and CPU, so system RAM is what bounds it)."""
    available, total = system_memory_bytes()
    return MemoryBudget(KIND_SYSTEM, "system memory", available, total, reason)
