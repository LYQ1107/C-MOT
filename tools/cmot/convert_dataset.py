"""Convert a supported local dataset or report an explicit unsupported status."""

import argparse
import json
from pathlib import Path

from cmot.data.bdd_converter import convert_bdd
from cmot.data.converters import BDD100KConverter, DatasetConverter, TAOConverter
from cmot.data.tao_converter import convert_tao_bdd
from cmot.manifest import write_json


SUPPORTED = {"bdd", "tao"}

# Keep the interface visible to downstream callers while only registering
# formats whose real local files were parsed in this execution.
CONVERTERS = {"bdd": BDD100KConverter, "tao": TAOConverter}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bdd", "tao", "ctao", "ovis", "vid", "vidor", "vidvrd", "gmot"), required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))
    output = Path(args.output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset not in SUPPORTED:
        print(json.dumps({"status": "UNSUPPORTED_DATASET_FORMAT", "dataset": args.dataset}, sort_keys=True))
        raise SystemExit(2)
    if args.dataset == "bdd":
        stats = convert_bdd(runtime["sources"], str(output), require_images=bool(args.validate))
    else:
        stats = convert_tao_bdd(runtime["tao_root"], str(output), require_images=bool(args.validate))
    print(json.dumps({"status": "OK", "stats": stats}, sort_keys=True))


if __name__ == "__main__":
    main()
