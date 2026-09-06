"""Audit or download explicitly declared sources through direct-only policy."""

import argparse
import json
from pathlib import Path

from cmot.config import load_config
from cmot.download import audit_local_route, direct_download
from cmot.manifest import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--download-policy", choices=("direct-only",), required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--download", action="store_true", help="download only sources whose route audit passes")
    args = parser.parse_args()
    config = load_config(args.config)
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))
    records = []
    for source in runtime.get("sources", []):
        url = source.get("url")
        if not url:
            records.append({"status": "BLOCKED_MISSING_URL", "name": source.get("name")})
            continue
        audit = audit_local_route(url)
        if not args.download:
            records.append({"name": source.get("name"), "status": "DRY_RUN", "audit": audit})
            continue
        output = source.get("output")
        if not output:
            records.append({"name": source.get("name"), "status": "BLOCKED_MISSING_OUTPUT", "audit": audit})
            continue
        records.append(direct_download(url, output, source.get("sha256")))
    value = {
        "schema_version": "cmot.download-manifest.v1",
        "policy": args.download_policy,
        "fallback_to_proxy": bool(config["network"]["fallback_to_proxy"]),
        "records": records,
    }
    write_json(args.manifest_output, value)
    print(json.dumps({"status": "OK", "records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
