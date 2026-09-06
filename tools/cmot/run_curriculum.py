"""Validate a curriculum request and expose its explicit target set."""

import argparse
import json
from pathlib import Path

from cmot.config import load_config
from cmot.manifest import write_json
from cmot.protocols.runner import CurriculumRunner, ExperimentTask, stable_experiment_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--max-concurrent-trainers", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="execute explicit runtime tasks; no implicit discovery/download")
    parser.add_argument("--output", help="runner status/summary JSON when --execute is used")
    args = parser.parse_args()
    config = load_config(args.config)
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))
    if args.max_concurrent_trainers < 1 or args.max_concurrent_trainers > int(config["resources"]["max_concurrent_trainers"]):
        raise SystemExit("requested concurrency exceeds resolved resource policy")
    targets = [value for value in args.targets.split(",") if value]
    allowed = {"asset_check", "smoke", "pilot"}
    if not set(targets) <= allowed:
        raise SystemExit("unknown curriculum target")
    if not args.execute:
        print(json.dumps({"status": "VALIDATED_NOT_DISPATCHED", "targets": targets, "max_concurrent_trainers": args.max_concurrent_trainers}, sort_keys=True))
        return
    tasks = runtime.get("tasks", [])
    if not tasks:
        raise SystemExit("--execute requires explicit runtime.tasks; no implicit stage dispatch is allowed")
    if args.max_concurrent_trainers != 1:
        raise SystemExit("only sequential execution is implemented by the bounded runner")
    results = []
    for raw in tasks:
        task = ExperimentTask(
            stage_id=raw["stage_id"],
            method=raw["method"],
            experiment_id=raw.get("experiment_id") or stable_experiment_id(
                raw["stage_id"], raw["method"], args.config,
                [raw.get("train_view"), raw.get("eval_view")],
                raw.get("parent_checkpoint"), raw.get("pl_manifest"),
                raw.get("memory_version"), int(config["seed"]), raw.get("budget", {}),
            ),
            output_dir=raw["output_dir"],
            expected_steps=int(raw["expected_steps"]),
            parent_checkpoint=raw.get("parent_checkpoint"),
            train_view=raw.get("train_view"),
            eval_view=raw.get("eval_view"),
            pl_manifest=raw.get("pl_manifest"),
            memory_version=raw.get("memory_version"),
            seed=int(config["seed"]),
            budget=raw.get("budget", {}),
        )
        runner = CurriculumRunner(task)
        train_kwargs = raw.get("train_kwargs")
        eval_kwargs = raw.get("eval_kwargs")
        if not isinstance(train_kwargs, dict) or not isinstance(eval_kwargs, dict):
            raise SystemExit("--execute tasks require explicit train_kwargs and eval_kwargs")

        def train_fn(current_task, values=dict(train_kwargs)):
            from cmot.train import train
            values = dict(values)
            values["output_dir"] = current_task.output_dir
            values["total_steps"] = current_task.expected_steps
            return train(**values)

        def eval_fn(current_task, train_result, values=dict(eval_kwargs)):
            from cmot.evaluate import evaluate_trackeval
            values = dict(values)
            checkpoint = values.pop("checkpoint", None)
            if checkpoint is not None and not Path(checkpoint).is_file():
                raise FileNotFoundError("evaluation checkpoint is missing")
            output = values.pop("output")
            return evaluate_trackeval(
                values.pop("eval_manifest"),
                values.pop("predictions"),
                output,
                values.pop("trackeval_root"),
                values.pop("max_videos", None),
                values.pop("max_frames_per_video", None),
            )

        results.append(runner.run(train_fn, eval_fn))
    value = {"status": "COMPLETE", "tasks": results}
    if args.output:
        write_json(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
