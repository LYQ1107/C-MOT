"""Manifest IO, hashes and public-safe metadata helpers."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def redact_path(value: str) -> str:
    """Replace local path roots without attempting to identify users."""
    value = str(value)
    for prefix, token in (
        ("/data1/", "$DATA1/"),
        ("/data2/", "$DATA2/"),
        ("/home/", "$HOME/"),
    ):
        if value.startswith(prefix):
            return token + value[len(prefix):]
    return value


def path_exists_and_size(path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": redact_path(path), "exists": os.path.exists(path)}
    if result["exists"]:
        stat = os.stat(path)
        result.update({"bytes": int(stat.st_size), "is_file": os.path.isfile(path), "is_dir": os.path.isdir(path)})
    return result

