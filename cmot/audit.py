"""Generate small private audit records without embedding server paths."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from .download import audit_local_route
from .manifest import redact_path, sha256_file, write_json


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


def runtime_record(config_path: str, **paths) -> dict:
    """Create a private runtime binding with redacted path values."""
    config = Path(config_path)
    result = {
        "schema_version": "cmot.runtime-local.v1",
        "config": {"basename": config.name, "sha256": sha256_file(str(config))},
        "credential_values_recorded": False,
        "parent_proxy_values_recorded": False,
        "paths": {key: redact_path(value) for key, value in paths.items() if value is not None},
    }
    return result


def write_network_audit(url: str, output: str) -> dict:
    value = audit_local_route(url)
    write_json(output, value)
    return value


def write_download_record(url: str, output: str) -> dict:
    # This audit command never performs a GET.  The actual downloader remains
    # available through cmot.download and is itself fail-closed.
    audit = audit_local_route(url)
    value = {
        "url": url,
        "status": "NOT_RUN" if audit.get("status") != "VERIFIED_DIRECT_ROUTE" else "NOT_RUN_HEAD_ONLY",
        "downloaded": False,
        "bytes": 0,
        "audit": audit,
    }
    jsonl_record(output, value)
    return value


def write_extraction_record(output: str, archive_name: str = "") -> dict:
    value = {
        "schema_version": "cmot.extraction-manifest.v1",
        "status": "NOT_RUN",
        "archive_basename": None if not archive_name else Path(archive_name).name,
        "bytes_extracted": 0,
        "credential_values_recorded": False,
        "reason": "no new archive passed the verified direct-route gate",
    }
    jsonl_record(output, value)
    return value


def write_experiment_layout(
    run_root: str,
    run_name: str,
    method: str,
    stage_id: str,
    scope: str,
    steps: int,
    config_path: str,
    train_view: str,
    eval_view: str,
    checkpoint_names,
    prediction_names,
    metric_names,
    parent_checkpoint_sha256: str,
    replay_status: str = "NOT_RUN",
) -> dict:
    """Create the required private run layout using links to existing artifacts."""
    source = Path(run_root) / run_name
    source.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "predictions", "raw_metrics"):
        (source / name).mkdir(exist_ok=True)

    def link(name: str, directory: str) -> None:
        original = source / name
        if not original.is_file():
            return
        destination = source / directory / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(os.path.relpath(str(original), str(destination.parent)))

    for name in checkpoint_names:
        link(name, "checkpoints")
    for name in prediction_names:
        link(name, "predictions")
    for name in metric_names:
        link(name, "raw_metrics")
    train_log = source / "train_steps.jsonl"
    if train_log.is_file():
        destination = source / "train_log.jsonl"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(os.path.relpath(str(train_log), str(destination.parent)))
    else:
        jsonl_record(str(source / "train_log.jsonl"), {"scope": scope, "optimizer_steps": 0, "status": "asset_check"})
    config_destination = source / "config_resolved.yaml"
    shutil.copy2(config_path, config_destination)
    manifest = {
        "schema_version": "cmot.experiment-manifest.v1",
        "experiment": run_name,
        "method": method,
        "stage_id": stage_id,
        "scope": scope,
        "optimizer_steps": int(steps),
        "config": {"basename": config_destination.name, "sha256": sha256_file(str(config_destination))},
        "train_view": Path(train_view).name if train_view else None,
        "eval_view": Path(eval_view).name if eval_view else None,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "checkpoint_basenames": [Path(name).name for name in checkpoint_names if (source / name).is_file()],
        "prediction_basenames": [Path(name).name for name in prediction_names if (source / name).is_file()],
        "raw_metric_basenames": [Path(name).name for name in metric_names if (source / name).is_file()],
        "replay": {"status": replay_status, "charged_from_existing_view": replay_status == "OK"},
        "credential_values_recorded": False,
    }
    write_json(str(source / "experiment_manifest.json"), manifest)
    write_json(str(source / "replay_manifest.json"), {
        "schema_version": "cmot.replay-manifest.v1",
        "status": replay_status,
        "source": "allowed_stage_view_only",
        "budget_mib": 512,
        "credential_values_recorded": False,
    })
    metric_exists = any((source / name).is_file() for name in metric_names)
    trained = scope == "asset_check" or any((source / name).is_file() for name in checkpoint_names)
    write_json(str(source / "status.json"), {
        "schema_version": "cmot.run-status.v1",
        "state": "COMPLETE" if trained and metric_exists else "NOT_RUN",
        "scope": scope,
        "optimizer_steps": int(steps),
        "raw_metrics_present": metric_exists,
        "checkpoint_present": trained,
    })
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root")
    parser.add_argument("--output")
    parser.add_argument("--runtime-output")
    parser.add_argument("--config")
    parser.add_argument("--runtime-path", action="append", default=[], help="logical_name=local_path")
    parser.add_argument("--network-url")
    parser.add_argument("--network-output")
    parser.add_argument("--download-url")
    parser.add_argument("--download-output")
    parser.add_argument("--extraction-output")
    parser.add_argument("--archive-name")
    args = parser.parse_args()
    if args.runtime_output:
        if not args.config:
            raise SystemExit("--runtime-output requires --config")
        paths = {}
        for item in args.runtime_path:
            if "=" not in item:
                raise SystemExit("--runtime-path must be logical_name=local_path")
            key, value = item.split("=", 1)
            paths[key] = value
        value = runtime_record(args.config, **paths)
        write_json(args.runtime_output, value)
        print(json.dumps({"status": "OK", "kind": "runtime"}, sort_keys=True))
        return
    if args.network_url or args.network_output:
        if not args.network_url or not args.network_output:
            raise SystemExit("--network-url and --network-output must be paired")
        value = write_network_audit(args.network_url, args.network_output)
        print(json.dumps({"status": value.get("status"), "kind": "network"}, sort_keys=True))
        return
    if args.download_url or args.download_output:
        if not args.download_url or not args.download_output:
            raise SystemExit("--download-url and --download-output must be paired")
        value = write_download_record(args.download_url, args.download_output)
        print(json.dumps({"status": value.get("status"), "kind": "download"}, sort_keys=True))
        return
    if args.extraction_output:
        value = write_extraction_record(args.extraction_output, args.archive_name or "")
        print(json.dumps({"status": value.get("status"), "kind": "extraction"}, sort_keys=True))
        return
    if not args.run_root or not args.output:
        raise SystemExit("checkpoint mode requires --run-root and --output")
    value = checkpoint_inventory(args.run_root)
    write_json(args.output, value)
    print(json.dumps({"status": "OK", "checkpoints": len(value["checkpoints"])}, sort_keys=True))


if __name__ == "__main__":
    main()
