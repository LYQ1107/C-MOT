"""Evaluate one explicit prediction file with local TrackEval."""

import argparse
import json
from pathlib import Path

from cmot.evaluate import evaluate_trackeval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-frames-per-video", type=int)
    args = parser.parse_args()
    if not Path(args.checkpoint).is_file():
        raise SystemExit("checkpoint does not exist: %s" % Path(args.checkpoint).name)
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))
    output = Path(args.output_dir) / "raw_metrics.json"
    result = evaluate_trackeval(args.eval_manifest, args.predictions, str(output), runtime["trackeval_root"], args.max_videos, args.max_frames_per_video)
    print(json.dumps({"status": result["status"], "output": output.name}, sort_keys=True))


if __name__ == "__main__":
    main()
