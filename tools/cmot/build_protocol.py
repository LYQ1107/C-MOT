"""Build named stage views from a canonical private manifest."""

import argparse
import json
from pathlib import Path

import yaml

from cmot.data.view_builder import build_stage_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))
    canonical = runtime["canonical"]
    output_root = Path(args.output_root)
    built = []
    stages = config.get("stages") or [
        {"stage_id": "S0_ref", "label_scope": "complete"},
        {"stage_id": "S1_pedestrian", "label_scope": "partial"},
        {"stage_id": "S2_truck", "label_scope": "partial"},
    ]
    pl_by_stage = runtime.get("pl_jsonl", {})
    for stage in stages:
        stage_id = stage["stage_id"]
        output = output_root / (stage_id + "_training_view.json")
        pl_jsonl = stage.get("pl_jsonl") or pl_by_stage.get(stage_id)
        build_stage_view(canonical, stage_id, str(output), pl_jsonl, int(stage.get("replay_max_frames", 128)), stage.get("split", "train"), "train", stage.get("label_scope"))
        built.append(output.name)
    print(json.dumps({"status": "OK", "views": built}, sort_keys=True))


if __name__ == "__main__":
    main()
