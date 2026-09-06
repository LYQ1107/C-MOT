"""Read-only inventory and provenance audit for a C-MOT workspace."""

import argparse
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .manifest import path_exists_and_size, redact_path, sha256_file, write_json


def _gpu_audit() -> dict:
    try:
        result = subprocess.run(
            ["/usr/bin/nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "protected_physical_gpus": [0, 1, 2, 3, 4], "allowed_physical_gpus": [5, 6, 7, 8, 9]}
    except Exception as exc:
        return {"error": type(exc).__name__ + ": " + str(exc), "protected_physical_gpus": [0, 1, 2, 3, 4], "allowed_physical_gpus": [5, 6, 7, 8, 9]}


def _asset(path: str, hash_file: bool = False) -> dict:
    result = path_exists_and_size(path)
    if hash_file and result.get("is_file"):
        result["sha256"] = sha256_file(path)
    return result


def build_inventory(project_root: str, asset_specs: Optional[Sequence[Tuple[str, str]]] = None) -> dict:
    """Build an inventory from operator-supplied paths.

    Paths are intentionally not embedded in the public source.  The server
    execution passes them at runtime and the JSON writer redacts their roots.
    """
    assets = {
        name: _asset(path, hash_file=Path(path).is_file())
        for name, path in (asset_specs or [])
    }
    return {
        "schema_version": "cmot.inventory.v1",
        "project_root": redact_path(project_root),
        "read_only_source_policy": True,
        "remote_source_not_modified": True,
        "assets": assets,
        "gpu_audit": _gpu_audit(),
        "environment": {
            "python_ovtr": "operator-selected",
            "python_evaluator": "operator-selected",
            "bulk_downloads_attempted": False,
            "credential_values_recorded": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset", action="append", default=[], help="logical_name=local_path (repeatable; paths are redacted in output)")
    args = parser.parse_args()
    specs = []
    for value in args.asset:
        if "=" not in value:
            raise SystemExit("--asset must be logical_name=local_path")
        name, path = value.split("=", 1)
        if not name or not path:
            raise SystemExit("--asset must contain a non-empty name and path")
        specs.append((name, path))
    output = build_inventory(args.project_root, specs)
    write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
