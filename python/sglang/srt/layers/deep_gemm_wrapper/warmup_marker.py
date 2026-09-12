import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_MARKER_DIRECTORY = ".sglang_warmup_complete"


def canonicalize_m_values(m_values: list[int]) -> list[int]:
    """Return the exact, stable M coverage represented by a marker."""
    return sorted({int(m) for m in m_values if int(m) > 0})


def make_marker_payload(
    *,
    identity: dict[str, Any],
    settings: dict[str, Any],
    kernel_type: str,
    n: int,
    k: int,
    num_groups: int,
    m_values: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "complete": True,
        "identity": identity,
        "settings": settings,
        "kernel": {
            "type": kernel_type,
            "n": int(n),
            "k": int(k),
            "num_groups": int(num_groups),
            "m_values": canonicalize_m_values(m_values),
        },
    }


def marker_path(cache_dir: str | Path, payload: dict[str, Any]) -> Path:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return Path(cache_dir) / _MARKER_DIRECTORY / f"{digest}.json"


def marker_matches(path: Path, expected_payload: dict[str, Any]) -> bool:
    """Require valid JSON and exact payload equality; every error is a miss."""
    try:
        with path.open(encoding="utf-8") as marker_file:
            actual_payload = json.load(marker_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return actual_payload == expected_payload


def write_marker_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Durably publish a completed marker without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as marker_file:
            temp_path = Path(marker_file.name)
            json.dump(payload, marker_file, sort_keys=True, separators=(",", ":"))
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.replace(temp_path, path)
        temp_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
