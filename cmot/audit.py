"""Generate small private audit records without embedding server paths."""

import argparse
import json
import re
from pathlib import Path

from .manifest import sha256_file, write_json


def checkpoint_inventory(run_root: str) -> dict:
    root = Path(run_root)
    records = []
    for path in sorted(root.rglob("*.pt")):
        if not path.is_file():
            continue
        match = re.search(r"checkpoint_(\d+)", path.name)
        records.append({
            "run": path.parent.name,
            "basename": path.name,
            "step": None if not match else int(match.group(1)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(str(path)),
        })
    return {
        "schema_version": "cmot.checkpoint-inventory.v1",
        "root_basename": root.name,
        "credential_values_recorded": False,
        "checkpoints": records,
    }


def jsonl_record(path: str, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = checkpoint_inventory(args.run_root)
    write_json(args.output, value)
    print(json.dumps({"status": "OK", "checkpoints": len(value["checkpoints"])}, sort_keys=True))


if __name__ == "__main__":
    main()
