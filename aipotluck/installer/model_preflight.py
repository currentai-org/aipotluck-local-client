"""Can this model physically exist on this device? Answered BEFORE anything is downloaded.

Every other check in this project costs a download first. `model_sizing` needs the model's own
hparams, which means loading it; `model_screen` needs to benchmark it. Both are the right tools for
"will this serve a turn well", and both arrive after the user has already spent ten minutes and
several gigabytes on a file that was never going to run.

This module answers the one question that can be settled from a file listing alone: are the weights
bigger than the machine? That is deliberately the crudest possible test, and it is chosen for its
false-positive rate rather than its coverage. A model whose weights exceed total installed memory
cannot be loaded at any context size, with any KV quantization, on any backend -- there is nowhere
to put it. Anything subtler (will there be room for a *useful* context? will it be fast enough?)
is left to the layers that measure, because guessing at those from a byte count would start
refusing models that work.

The size comes from the same place llama.cpp's own `-hf` downloader gets it: Hugging Face's
`api/models/{repo}/tree/{ref}` listing. The quant-matching below mirrors `find_best_model` and
`get_gguf_split_info` in vendor/llama.cpp/common/download.cpp, so the file this module measures is
the file the downloader would fetch -- including sharded models, where every shard counts toward
what has to be resident.

Nothing here is authoritative enough to fail a pull on its own. Every failure path returns None,
meaning "could not tell": a private repo, a rate limit, an offline machine or an unfamiliar
filename all produce no verdict rather than a wrong one.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("aipotluck.installer.model_preflight")

_HF_ENDPOINT = "https://huggingface.co"
_DEFAULT_TIMEOUT_SECONDS = 15.0

# When no quant is named, llama.cpp tries these in order (download.cpp's find_best_model).
_DEFAULT_TAGS = ("Q4_K_M", "Q8_0")

# Sidecar GGUFs that are not the model itself -- a projector or a draft model gets downloaded
# alongside, but it is not what has to fit. Mirrors gguf_filename_is_model.
_NON_MODEL_MARKERS = ("mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-")

_SPLIT_RE = re.compile(r"^(.+)-([0-9]{5})-of-([0-9]{5})$", re.IGNORECASE)

# Environment variables Hugging Face's own tooling reads, in its order of preference. A token only
# matters for private or gated repos; public ones resolve anonymously.
_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


@dataclass
class RemoteModelSize:
    """What the downloader would actually fetch for a given `repo:quant`."""

    total_bytes: int
    primary_path: str
    paths: list[str] = field(default_factory=list)


@dataclass
class PreflightVerdict:
    fits: bool
    model_bytes: int
    total_memory_bytes: int
    detail: str


def _token() -> str | None:
    for name in _TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def split_repo_tag(model_id: str) -> tuple[str, str]:
    """`org/repo:Q4_K_M` -> ("org/repo", "Q4_K_M"); a bare `org/repo` -> ("org/repo", "")."""
    repo, _, tag = model_id.partition(":")
    return repo, tag


def _is_model_file(path: str) -> bool:
    if not path.endswith(".gguf"):
        return False
    filename = path.rsplit("/", 1)[-1]
    return not any(marker in filename for marker in _NON_MODEL_MARKERS)


def _split_info(path: str) -> tuple[str, int, int]:
    """(prefix, index, count) for a possibly-sharded GGUF. An unsharded file is (name, 1, 1)."""
    if not path.endswith(".gguf"):
        return "", 1, 1
    prefix = path[: -len(".gguf")]
    match = _SPLIT_RE.match(prefix)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return prefix, 1, 1


def _fetch_tree(repo: str, *, timeout: float) -> list[dict] | None:
    """The repository's file listing, or None if it can't be read for any reason."""
    url = (
        f"{_HF_ENDPOINT}/api/models/{urllib.parse.quote(repo)}"
        "/tree/main?recursive=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "aipotluck-local-client"})
    token = _token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        log.debug("Could not list %s on Hugging Face: %s", repo, exc)
        return None
    if not isinstance(payload, list):
        return None
    return payload


def remote_model_bytes(model_id: str, *, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> RemoteModelSize | None:
    """Total bytes the `-hf` downloader would fetch for `model_id`, or None if it can't be told.

    Resolution deliberately mirrors llama.cpp's: match the quant tag against the filename as
    `tag[.-]` case-insensitively, ignore sidecar GGUFs, and for a sharded model sum every shard
    rather than reporting the first one -- all of them have to be resident to serve a token."""
    repo, tag = split_repo_tag(model_id)
    if "/" not in repo:
        return None

    entries = _fetch_tree(repo, timeout=timeout)
    if not entries:
        return None

    files = {
        entry["path"]: int(entry.get("size") or 0)
        for entry in entries
        if isinstance(entry, dict) and entry.get("type") == "file" and entry.get("path")
    }
    model_files = [path for path in files if _is_model_file(path)]
    if not model_files:
        return None

    for candidate_tag in ([tag] if tag else list(_DEFAULT_TAGS)):
        pattern = re.compile(re.escape(candidate_tag) + r"[.\-]", re.IGNORECASE)
        for path in model_files:
            if not pattern.search(path):
                continue
            prefix, index, count = _split_info(path)
            if count > 1 and index != 1:
                continue  # not the head of the shard set; its siblings are summed below
            if count > 1:
                shards = [
                    other for other in model_files
                    if _split_info(other)[0] == prefix and _split_info(other)[2] == count
                ]
            else:
                shards = [path]
            total = sum(files[shard] for shard in shards)
            if total <= 0:
                return None
            return RemoteModelSize(total_bytes=total, primary_path=path, paths=sorted(shards))
    return None


def check_fits_in_memory(
    model_id: str,
    *,
    total_memory_gb: float | None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> PreflightVerdict | None:
    """Whether `model_id`'s weights alone exceed this machine's total memory.

    None means no verdict -- the size couldn't be read, or memory couldn't be measured. That is a
    deliberate outcome rather than a failure: this gate exists to stop a certainly-doomed download,
    so the absence of evidence must let the pull proceed to the layers that measure for real.

    The bar is TOTAL installed memory, not the 80% budget model_sizing allocates. Those answer
    different questions. The budget asks "will this run well", and overshooting it is a reason to
    size conservatively, not to refuse. This asks "can this exist", and the honest threshold for
    that is the whole machine -- including the part the OS is using, which the user could in
    principle free. Weights larger than all of it cannot be loaded by any configuration."""
    if total_memory_gb is None:
        return None
    size = remote_model_bytes(model_id, timeout=timeout)
    if size is None:
        return None

    total_memory_bytes = int(total_memory_gb * (1024**3))
    model_gb = size.total_bytes / (1024**3)
    shard_note = f" across {len(size.paths)} files" if len(size.paths) > 1 else ""

    if size.total_bytes > total_memory_bytes:
        detail = (
            f"{model_id} is {model_gb:.1f}GB of weights{shard_note}, and this device has "
            f"{total_memory_gb:.1f}GB of memory in total. The weights alone do not fit, so it "
            "cannot be loaded at any context size or quantization."
        )
        return PreflightVerdict(False, size.total_bytes, total_memory_bytes, detail)

    detail = (
        f"{model_id} is {model_gb:.1f}GB of weights{shard_note}, within this device's "
        f"{total_memory_gb:.1f}GB of memory."
    )
    return PreflightVerdict(True, size.total_bytes, total_memory_bytes, detail)
