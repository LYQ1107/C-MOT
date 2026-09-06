"""Spec-compatible wrapper for the public report generator."""

import argparse
import json

from cmot.summarize import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--public-output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--require-raw-evidence", action="store_true")
    args = parser.parse_args()
    value = build_report(args.run_root, args.project_root, args.public_output)
    if args.require_raw_evidence and any(row["status"] == "OK" and not row["raw_metric_sha256"] for row in value["rows"]):
        raise SystemExit("raw evidence missing")
    print(json.dumps({"status": "OK", "rows": len(value["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
