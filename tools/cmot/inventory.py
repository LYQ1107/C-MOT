"""Spec-compatible inventory command."""

import argparse
import json

from cmot.inventory import build_inventory
from cmot.manifest import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-root", default=".")
    parser.add_argument("--asset", action="append", default=[], help="logical_name=local_path")
    args = parser.parse_args()
    output = args.output_dir.rstrip("/") + "/resource_inventory.json"
    specs = []
    for item in args.asset:
        if "=" not in item:
            raise SystemExit("--asset must be logical_name=local_path")
        specs.append(tuple(item.split("=", 1)))
    value = build_inventory(args.candidate_root, specs)
    write_json(output, value)
    print(json.dumps({"status": "OK", "output": "resource_inventory.json"}, sort_keys=True))


if __name__ == "__main__":
    main()
