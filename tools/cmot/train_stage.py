"""Run one explicit C-MOT stage from a private resolved runtime binding."""

import argparse
import json
from pathlib import Path

from cmot.config import load_config
from cmot.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init-weights")
    group.add_argument("--resume")
    parser.add_argument("--max-optimizer-steps", type=int, required=True)
    args = parser.parse_args()
    load_config(args.config)
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))
    values = dict(runtime["train"])
    values["total_steps"] = args.max_optimizer_steps
    values["checkpoint_init"] = args.init_weights or values.get("checkpoint_init")
    values["resume"] = args.resume
    if not values.get("checkpoint_init"):
        raise SystemExit("runtime train.checkpoint_init is required with --resume")
    result = train(**values)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
