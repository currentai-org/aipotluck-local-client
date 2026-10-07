"""Read/write for the router-mode preset file (`llama-server --models-preset`) -- the mechanical
half of CUR-1965's auto-sizing follow-up; `model_sizing.py` decides *what* values go in, this
module only knows *how* to get them in and out of the INI file llama-server itself reads.

Confirmed against vendor/llama.cpp/common/preset.cpp's real parser (a hand-rolled PEG grammar, not
Python's configparser -- we don't need to match its grammar exactly, only produce output it accepts):
a section header is any text up to `]`, matched directly against a model's identity (for a
cache-discovered model, the exact `repo:quant` string `--cache-list`/`model_pull.list_cached_models`
already use -- verified via `common_preset_context::load_from_cache()`, which names each cache entry
`model.to_string()`, the same call `--cache-list` prints); each key is the option's long-form CLI
flag spelling with leading dashes stripped (`--ctx-size` -> `ctx-size`), and both short and long
spellings resolve to the same option, so there's no ambiguity in always writing the long form here.

Every other section in the file -- another model, the global `[*]` section, anything a person
hand-edited -- is preserved verbatim on every write. This is always a read-modify-write over the
whole file, never a regenerate-from-scratch, since a user is free to add their own preset entries
for models this project doesn't manage.
"""

from __future__ import annotations

import configparser
import json
from pathlib import Path
from typing import Any

# llama.cpp's own convention for "applies to every preset unless overridden" -- never a real
# model id, so it's excluded from anything that lists "the models this project has sized".
_GLOBAL_SECTION = "*"


def _read(ini_path: Path) -> configparser.ConfigParser:
    # optionxform=str: preserve key spelling/case exactly. configparser lowercases option names by
    # default, which would silently rewrite anything hand-edited with different casing; llama.cpp's
    # own keys are already all-lowercase-with-dashes so this is a no-op for us, but it keeps a
    # human's untouched sections genuinely untouched. interpolation=None: preset values like
    # "q8_0" never need it, and it's one less way a value could be silently mis-parsed.
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # type: ignore[method-assign]
    if ini_path.exists():
        parser.read(ini_path, encoding="utf-8")
    return parser


def has_preset(ini_path: Path, model_id: str) -> bool:
    if not ini_path.exists():
        return False
    return _read(ini_path).has_section(model_id)


def write_preset(ini_path: Path, model_id: str, args: dict[str, str | None]) -> None:
    """Sets `args` (e.g. {"ctx-size": "16384", "cache-type-k": "q8_0"}) on model_id's section,
    creating it if needed. A None value removes that key (used when a model can't use a quantized
    KV cache, so cache-type-k/v should be absent -- llama-server's own f16 default then applies --
    rather than ever writing a placeholder). Every other section is left exactly as it was."""
    ini_path.parent.mkdir(parents=True, exist_ok=True)
    parser = _read(ini_path)
    if not parser.has_section(model_id):
        parser.add_section(model_id)
    for key, value in args.items():
        if value is None:
            parser.remove_option(model_id, key)
        else:
            parser.set(model_id, key, str(value))
    with ini_path.open("w", encoding="utf-8") as fh:
        parser.write(fh)


def known_model_ids(ini_path: Path) -> set[str]:
    """Every model this project has already written a preset section for -- used by the
    service-startup backfill to know what NOT to touch, and by `list` to annotate sizing status."""
    if not ini_path.exists():
        return set()
    return set(_read(ini_path).sections()) - {_GLOBAL_SECTION}


def read_all(ini_path: Path) -> dict[str, dict[str, str]]:
    """Every managed model's raw key/value pairs, keyed by model id -- e.g.
    {"org/repo:Q4_K_M": {"ctx-size": "16384", "cache-type-k": "q8_0", ...}}. Used by
    aipotluck.diagnostics.runtime_params to build the traceable, per-model view `status`/
    `/capabilities` show; never includes the `[*]` global section (see `known_model_ids`)."""
    if not ini_path.exists():
        return {}
    parser = _read(ini_path)
    return {section: dict(parser.items(section)) for section in parser.sections() if section != _GLOBAL_SECTION}


def _tuning_path(ini_path: Path) -> Path:
    return ini_path.with_suffix(".tuning.json")


def read_tuning(ini_path: Path) -> dict[str, dict[str, str]]:
    """The `{param: reason}` traceability map for every sized model, keyed by model id (see
    CLAUDE.md's "Runtime parameters" convention). Kept as a sibling JSON file rather than INI
    comments -- llama-server's own preset parser never reads this file, it's purely for `status`/
    `/capabilities` to explain a preset's numbers, so it doesn't need to round-trip through
    llama.cpp's grammar at all."""
    path = _tuning_path(ini_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_tuning(ini_path: Path, model_id: str, tuning: dict[str, str]) -> None:
    """Read-modify-write, same posture as write_preset -- another model's tuning reasons are
    never touched by a write for this one."""
    path = _tuning_path(ini_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = read_tuning(ini_path)
    data[model_id] = tuning
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def remove_preset(ini_path: Path, model_id: str) -> bool:
    """Drops model_id's section and its tuning reasons, for a model whose files are gone. Left
    behind, a section names a model the router can no longer serve, and `status` would keep
    explaining the sizing of something that isn't there. Every other section is preserved, the
    same as write_preset. Returns whether there was a section to drop."""
    removed = False
    if ini_path.exists():
        parser = _read(ini_path)
        if parser.remove_section(model_id):
            with ini_path.open("w", encoding="utf-8") as fh:
                parser.write(fh)
            removed = True
    tuning = read_tuning(ini_path)
    if tuning.pop(model_id, None) is not None:
        _tuning_path(ini_path).write_text(json.dumps(tuning, indent=2), encoding="utf-8")
    return removed
