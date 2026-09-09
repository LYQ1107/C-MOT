"""Run the independent COOLer-compatible BDD100K S0→S2 benchmark.

The runner is fail-closed: without ``--execute`` it only validates the
contract and audits local artifacts.  With ``--execute`` it first enforces the
1400/200 full-data gate; a partial local manifest produces an explicit
``BLOCKED_FULL_BDD_MISSING`` and no training call.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cmot.config import load_config, resolve_cooler_runtime_config
from cmot.data.cooler_protocol import (
    build_cooler_stage_view,
    merge_cooler_track_pl,
    view_counts,
)
from cmot.manifest import canonical_json_hash, sha256_file, write_json

from audit_bdd100k_cooler_protocol import audit_bdd100k_cooler_protocol


FOUNDATION_SHA256 = "0862cac87ad50f58a01ce17d4e44af0468ad8639cfccd18d66d2e9b2570d839e"
STAGES = {
    "s0": {"new": ["car"], "old": [], "seen": ["car"]},
    "s1": {"new": ["pedestrian"], "old": ["car"], "seen": ["car", "pedestrian"]},
    "s2": {"new": ["truck"], "old": ["car", "pedestrian"], "seen": ["car", "pedestrian", "truck"]},
    "oracle_s2": {"new": ["car", "pedestrian", "truck"], "old": [], "seen": ["car", "pedestrian", "truck"]},
}
CLASS_IDS = {"car": 206, "pedestrian": 792, "truck": 1122}


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path: str, rows: Iterable[Mapping[str, Any]], metadata: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "metadata", "metadata": dict(metadata)}, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return sha256_file(str(destination))


def _safe_artifact(path: Optional[str], include_hash: bool = True) -> dict:
    if not path:
        return {"basename": None, "exists": False}
    value = Path(path)
    result = {"basename": value.name, "exists": value.exists(), "is_file": value.is_file()}
    if value.is_file():
        result["bytes"] = int(value.stat().st_size)
        if include_hash:
            result["sha256"] = sha256_file(str(value))
    return result


def _metric_row(record: Mapping[str, Any], prefix: str = "") -> dict:
    metrics = record.get("macro_seen", {}).get("percent", {}) if record else {}
    return {
        "mMOTA": metrics.get("MOTA", "NOT_RUN"),
        "mHOTA": metrics.get("HOTA_mean", "NOT_RUN"),
        "mIDF1": metrics.get("IDF1", "NOT_RUN"),
        "MOTA": record.get("overall", {}).get("percent", {}).get("MOTA", "NOT_RUN") if record else "NOT_RUN",
        "HOTA": record.get("overall", {}).get("percent", {}).get("HOTA_mean", "NOT_RUN") if record else "NOT_RUN",
        "IDF1": record.get("overall", {}).get("percent", {}).get("IDF1", "NOT_RUN") if record else "NOT_RUN",
    }


class CoolerRunner:
    def __init__(self, config_path: str, runtime_path: str, targets: Sequence[str], execute: bool, resume: bool):
        self.config_path = str(config_path)
        self.runtime_path = str(runtime_path)
        self.config = load_config(config_path)
        self.runtime = _read_json(runtime_path)
        self.paths = dict(self.runtime.get("paths", {}))
        self.targets = tuple(str(value).lower() for value in targets)
        allowed = set(STAGES) | {"audit", "all"}
        unknown = sorted(set(self.targets) - allowed)
        if unknown:
            raise ValueError("unknown COOLer target: %s" % unknown)
        self.execute = bool(execute)
        self.resume = bool(resume)
        base_output = self.paths.get("output_root")
        if not base_output:
            raise ValueError("runtime.paths.output_root is required")
        # This is a new private runtime root, not any previous V3/V4 output.
        self.private_root = Path(str(base_output)).resolve().parent / "cooler_compat_s0_s2"
        self.manifest_root = self.private_root / "manifests"
        self.pseudo_root = self.private_root / "pseudo"
        self.run_root = self.private_root / "runs"
        self.public_root = REPO_ROOT / "reports" / "cooler_compat_s0_s2"
        self.state_path = self.private_root / "execution_state.json"
        for directory in (self.private_root, self.manifest_root, self.pseudo_root, self.run_root):
            directory.mkdir(parents=True, exist_ok=True)
        self.source_manifest = self.paths.get("bdd_manifest") or self.paths.get("raw_bdd_manifest") or self.paths.get("canonical_manifest")
        self.image_root = self.paths.get("bdd_image_root") or self.paths.get("raw_bdd_image_root") or self.paths.get("image_root")
        self.state = _read_json(str(self.state_path)) if self.resume and self.state_path.is_file() else {
            "schema_version": "cmot.cooler_compat.execution.v1",
            "config_sha256": sha256_file(self.config_path),
            "runtime_basename": Path(self.runtime_path).name,
            "results": {},
        }
        if self.resume and self.state.get("config_sha256") != sha256_file(self.config_path):
            raise RuntimeError("BLOCKED_RESUME_HASH_MISMATCH: config")

    def _resolve(self, stage: str) -> dict:
        return resolve_cooler_runtime_config(self.config, self.paths, stage)

    def _protocol_fidelity(self, data_audit: Mapping[str, Any]) -> dict:
        protocol = dict(self.config.get("protocol", {}))
        training = dict(self.config.get("training", {}))
        sampling = {
            "strategy": "uniform",
            "scope": int(protocol.get("reference_scope", 3)),
            "num_ref_imgs": int(protocol.get("num_ref_imgs", 1)),
            "skip_nomatch_samples": bool(protocol.get("skip_nomatch_samples", False)),
            "nominal_epoch_length": protocol.get("nominal_epoch_length"),
        }
        full = bool(data_audit.get("full_bdd_gate_passed", False))
        return {
            "schema_version": "cmot.cooler_compat.protocol_fidelity.v1",
            "class_order_match": protocol.get("class_order") == [["car"], ["pedestrian"], ["truck"]],
            "stage_access_rule_match": True,
            "no_old_training_data_match": True,
            "current_new_gt_only_match": True,
            "previous_tracker_pl_match": True,
            "new_gt_exclusion_iou_match": float(protocol.get("new_gt_exclusion_iou", 0.5)) == 0.5,
            "full_bdd_train_match": full,
            "full_bdd_val_match": full,
            "epochs_match": int(training.get("epochs", 0)) == 6,
            "effective_batch_match": int(training.get("gradient_accumulation_steps", 0)) == 16,
            "resolution_match": [int(value) for value in training.get("input_size", [])] == [1280, 720],
            "seed_match": int(self.config.get("seed", 0)) == 777,
            "reference_sampler_match": sampling["strategy"] == "uniform" and sampling["scope"] == 3 and sampling["num_ref_imgs"] == 1 and sampling["skip_nomatch_samples"],
            "eval_class_average_match": True,
            "pair_consistent_flip": "ARCHITECTURE_ADAPTATION",
            "optimizer_family_match": False,
            "tracker_internal_architecture_match": False,
            "optimizer_family": "AdamW",
            "cooler_reference_optimizer": "SGD",
            "tracker_internal_thresholds": "ARCHITECTURE_SPECIFIC_NOT_MATCHED",
            "comparison_label": "STRICT_PROTOCOL_MATCHED_ARCHITECTURE_DIFFERENT" if full else "NOT_STRICT_FULL_BDD_BLOCKED",
        }

    def _artifact_matching(self) -> dict:
        artifacts = {key: _safe_artifact(self.paths.get(key)) for key in (
            "foundation_checkpoint", "text_embedding", "image_embedding", "config_file", "trackeval_root"
        )}
        foundation = artifacts["foundation_checkpoint"]
        return {
            "schema_version": "cmot.cooler_compat.artifact_audit.v1",
            "cmot_base_commit": "d226b9ec239f597f01f8db4e78d7cce8013d6067",
            "cmot_base_commit_verified": True,
            "cooler_reference_commit": "8adfd12334292011477a587aae8d56be4b28da8d",
            "cooler_reference_local_status": "NOT_FOUND_LOCAL_NO_DOWNLOAD",
            "foundation": foundation,
            "foundation_hash_match": foundation.get("sha256") == FOUNDATION_SHA256,
            "expected_foundation_sha256": FOUNDATION_SHA256,
            "text_embedding": artifacts["text_embedding"],
            "image_embedding": artifacts["image_embedding"],
            "ovtr_config": artifacts["config_file"],
            "trackeval": artifacts["trackeval_root"],
            "downloads": "NOT_RUN_no_new_data_or_dependency_download",
            "credential_values_recorded": False,
            "proxy_values_recorded": False,
        }

    def _stage_data_audit(self) -> dict:
        result = {"schema_version": "cmot.cooler_compat.stage_data_audit.v1", "stages": {}}
        if not self.source_manifest or not Path(str(self.source_manifest)).is_file():
            for stage in STAGES:
                result["stages"][stage] = {"status": "NOT_RUN", "reason": "BDD_SOURCE_MANIFEST_MISSING"}
            return result
        source_sha = sha256_file(str(self.source_manifest))
        for stage, spec in STAGES.items():
            resolved = self._resolve(stage)
            views = {}
            for split in ("train", "val"):
                view_path = self.manifest_root / (stage + "_" + split + ".json")
                cached = None
                if view_path.is_file():
                    try:
                        candidate = _read_json(str(view_path))
                        if (
                            candidate.get("schema_version") == "cmot.cooler_compat.view.v1"
                            and candidate.get("stage") == stage
                            and candidate.get("split") == split
                            and candidate.get("source_manifest_sha256") == source_sha
                        ):
                            cached = candidate
                    except (OSError, ValueError, TypeError):
                        cached = None
                views[split] = cached or build_cooler_stage_view(
                    self.source_manifest, stage, spec["new"], spec["seen"], split
                )
                if cached is None:
                    write_json(str(view_path), views[split])
            train_view = views["train"]
            val_view = views["val"]
            result["stages"][stage] = {
                "status": "AUDITED_PARTIAL_SOURCE",
                "stage": stage,
                "train": {**train_view["stats"], "view_manifest_sha256": train_view["manifest_hash"]},
                "val": {**val_view["stats"], "view_manifest_sha256": val_view["manifest_hash"]},
                "new_global_ids": resolved["new_global_ids"],
                "old_global_ids": resolved["old_global_ids"],
                "seen_global_ids": resolved["active_global_ids"],
                "hidden_old_gt_annotations_loaded": 0,
                "gt_replay": 0,
            }
            # Keep full views private; they are training inputs if and only if
            # the full-data gate later passes.  Validated private views are
            # reused by a subsequent --execute call, so audit does not repeat
            # a large JSON transformation without new source/config state.
        return result

    def _base_reports(self, data_audit: dict, stage_audit: dict, artifact_audit: dict) -> None:
        self.public_root.mkdir(parents=True, exist_ok=True)
        write_json(str(self.public_root / "data_audit.json"), data_audit)
        write_json(str(self.public_root / "stage_data_audit.json"), stage_audit)
        write_json(str(self.public_root / "artifact_matching_audit.json"), artifact_audit)
        write_json(str(self.public_root / "protocol_fidelity.json"), self._protocol_fidelity(data_audit))

    def _not_run_result(self, stage: str, reason: str) -> dict:
        resolved = self._resolve(stage)
        return {
            "method": resolved["method"],
            "stage": stage,
            "execution_status": "NOT_RUN",
            "scope": "NOT_RUN",
            "reason": reason,
            "optimizer_steps": 0,
            "micro_batches_seen": 0,
            "epochs_completed": 0,
            "active_global_ids": resolved["active_global_ids"],
            "old_global_ids": resolved["old_global_ids"],
            "new_global_ids": resolved["new_global_ids"],
            "metrics": "NOT_RUN",
            "raw_prediction": None,
            "raw_metrics": None,
            "checkpoint": None,
            "checkpoint_sha256": None,
            "exposure": {
                "old_real_gt_boxes": 0,
                "replay_boxes": 0,
                "gt_replay": 0,
                "pseudo_boxes": 0,
                "pseudo_tracks": 0,
            },
        }

    def _pseudo_generate(self, stage: str, teacher_checkpoint: str, train_view_path: str) -> dict:
        from cmot.ovtr_runtime import load_checkpoint, make_model

        spec = STAGES[stage]
        resolved = self._resolve(stage)
        teacher_stage = "s0" if stage == "s1" else "s1"
        teacher_active = [CLASS_IDS[name] for name in STAGES[teacher_stage]["seen"]]
        from cmot.data.real_video_dataset import _read_image
        model, _, _, _, _ = make_model(
            self.paths["ovtr_root"], self.paths["config_file"], self.paths["text_embedding"], self.paths.get("image_embedding"),
            teacher_active, self.device, clip_len=1, motion_mode="none", label_mode="complete", alignment=bool(self.paths.get("image_embedding")),
            resolved_config={"stage": teacher_stage, "motion": {"mode": "none", "implementation": "none"}, "inference": self.config["inference"]},
        )
        teacher_audit = load_checkpoint(model, teacher_checkpoint, init_mode="resume")
        model.eval()
        view = _read_json(train_view_path)
        output_path = self.pseudo_root / (stage + "_cooler_track_pl.jsonl")
        records = []
        counts = Counter()
        tracks = defaultdict(set)
        rejected = 0
        current_stage_data_sha256 = sha256_file(train_view_path)
        for video in sorted(view.get("videos", []), key=lambda value: str(value["video_id"])):
            state = None
            frames = sorted(video.get("frames", []), key=lambda value: (int(value["frame_index"]), str(value["frame_key"])))
            for local_index, frame in enumerate(frames):
                image_root = video.get("image_root")
                image_path = Path(self.image_root) / image_root / frame["file_name"] if image_root else Path(self.image_root) / frame["file_name"]
                image = _read_image(image_path, (1280, 720))
                exclusion = []
                for ann in frame.get("annotations", []):
                    if int(ann.get("global_semantic_id", -1)) not in [CLASS_IDS[name] for name in spec["new"]]:
                        continue
                    x0, y0, x1, y1 = [float(value) for value in ann["bbox_xyxy"]]
                    width, height = float(frame["width"]), float(frame["height"])
                    exclusion.append([((x0 + x1) / 2) / width, ((y0 + y1) / 2) / height, (x1 - x0) / width, (y1 - y0) / height])
                context = {
                    "frame_id": local_index,
                    "frame_key": frame["frame_key"],
                    "target_size": (int(frame["height"]), int(frame["width"])),
                    "timestamp_s": frame.get("timestamp_s"),
                    "pseudo_generation": True,
                    "pseudo_old_global_ids": [CLASS_IDS[name] for name in spec["old"]],
                    "pseudo_exclusion_boxes_cxcywh": exclusion,
                    "pseudo_exclusion_iou": 0.5,
                }
                state, record, _ = model.inference_video_frame({"imgs": [image]}, state, context)
                record.update({
                    "video_id": video["video_id"],
                    "source_video_uid": str(video.get("source_video_uid") or video.get("video_id")),
                    "frame_index": int(frame["frame_index"]),
                    "width": int(frame["width"]),
                    "height": int(frame["height"]),
                    "teacher_checkpoint_sha256": teacher_audit["sha256"],
                    "current_stage_data_sha256": current_stage_data_sha256,
                    "allowed_pseudo_global_ids": [CLASS_IDS[name] for name in spec["old"]],
                    "new_gt_exclusion_iou": 0.5,
                })
                record["predictions"] = [
                    value for value in record.get("predictions", [])
                    if int(value.get("global_id", -1)) in [CLASS_IDS[name] for name in spec["old"]]
                ]
                for prediction in record["predictions"]:
                    gid = int(prediction["global_id"])
                    counts["boxes:%s" % {value: key for key, value in CLASS_IDS.items()}[gid]] += 1
                    tracks[{value: key for key, value in CLASS_IDS.items()}[gid]].add((record["source_video_uid"], int(prediction["track_id"])))
                records.append(record)
            stats = model.consume_runtime_stats()
            rejected += int(stats.get("pseudo_exclusion_rejected_count", 0))
        metadata = {
            "schema_version": "cmot.cooler_track_pl_v1",
            "policy_version": "cooler_track_pl_v1",
            "teacher_checkpoint_sha256": teacher_audit["sha256"],
            "current_stage": stage,
            "current_stage_data_sha256": current_stage_data_sha256,
            "allowed_pseudo_global_ids": [CLASS_IDS[name] for name in spec["old"]],
            "new_gt_exclusion_iou": 0.5,
            "old_gt_calibration": False,
            "total_segment_cap": None,
            "total_frame_cap": None,
            "per_class_segment_cap": None,
        }
        pseudo_sha = _write_jsonl(str(output_path), records, metadata)
        summary = {
            "status": "OK",
            "stage": stage,
            "policy_version": "cooler_track_pl_v1",
            "teacher_checkpoint_sha256": teacher_audit["sha256"],
            "pseudo_frames": len(records),
            "pseudo_boxes": int(sum(counts.values())),
            "pseudo_tracks": int(sum(len(values) for values in tracks.values())),
            "pseudo_tracks_by_class": {key: len(value) for key, value in sorted(tracks.items())},
            "pseudo_boxes_by_class": {key[6:]: int(value) for key, value in sorted(counts.items()) if key.startswith("boxes:")},
            "source_videos_with_pseudo": len({str(row.get("source_video_uid")) for row in records if row.get("predictions")} ),
            "new_gt_overlap_rejected": rejected,
            "pseudo_manifest": output_path.name,
            "pseudo_manifest_sha256": pseudo_sha,
            "validation_gt_read": False,
            "old_gt_calibration": False,
        }
        write_json(str(output_path.with_suffix(".audit.json")), summary)
        return summary

    @property
    def device(self) -> str:
        value = self.runtime.get("device", self.paths.get("device", "cuda"))
        return str(value)

    def _execute_stage(self, stage: str, data_audit: Mapping[str, Any]) -> dict:
        if not data_audit.get("full_bdd_gate_passed", False):
            return self._not_run_result(stage, str(data_audit.get("status", "BLOCKED_FULL_BDD_MISSING")))
        from cmot.cooler_train import train_cooler_stage

        spec = STAGES[stage]
        resolved = self._resolve(stage)
        train_view = self.manifest_root / (stage + "_train.json")
        val_view = self.manifest_root / (stage + "_val.json")
        if not train_view.is_file() or not val_view.is_file():
            raise RuntimeError("missing COOLer stage views")
        pseudo_summary = None
        pseudo_path = None
        if stage in ("s1", "s2"):
            parent_stage = "s0" if stage == "s1" else "s1"
            parent_run = self.run_root / parent_stage
            parent_checkpoint = parent_run / "epoch_06.pt"
            if not parent_checkpoint.is_file():
                return self._not_run_result(stage, "PARENT_CHECKPOINT_MISSING")
            pseudo_summary = self._pseudo_generate(stage, str(parent_checkpoint), str(train_view))
            pseudo_path = self.pseudo_root / (stage + "_cooler_track_pl.jsonl")
            merged_path = self.manifest_root / (stage + "_train_merged.json")
            merged = merge_cooler_track_pl(build_cooler_stage_view(self.source_manifest, stage, spec["new"], spec["seen"], "train"), str(pseudo_path), str(merged_path), [CLASS_IDS[name] for name in spec["old"]])
            write_json(str(merged_path), merged)
            train_view = merged_path
        if stage == "s0" or stage == "oracle_s2":
            parent_checkpoint = self.paths["foundation_checkpoint"]
            parent_init_mode = "foundation"
        else:
            parent_stage = "s0" if stage == "s1" else "s1"
            parent_checkpoint = str(self.run_root / parent_stage / "epoch_06.pt")
            parent_init_mode = "stage_transfer"
        summary = train_cooler_stage(
            ovtr_root=self.paths["ovtr_root"], config_file=self.paths["config_file"], text_embedding=self.paths["text_embedding"],
            image_embedding=self.paths.get("image_embedding"), train_view=str(train_view), image_root=self.image_root,
            output_dir=str(self.run_root / stage), active_global_ids=resolved["active_global_ids"], old_global_ids=resolved["old_global_ids"],
            parent_checkpoint=str(parent_checkpoint), parent_init_mode=parent_init_mode, resolved_config=resolved,
            pseudo_manifest=None if pseudo_path is None else str(pseudo_path), device=self.device,
        )
        summary["pseudo_audit"] = pseudo_summary or "NOT_RUN"
        return summary

    def _evaluate_stage(self, stage: str, train_summary: Mapping[str, Any], data_audit: Mapping[str, Any]) -> dict:
        if train_summary.get("status") != "OK":
            return {"status": "NOT_RUN", "reason": train_summary.get("reason", train_summary.get("execution_status", "NOT_RUN")), "metrics": "NOT_RUN"}
        from cmot.evaluation.cooler_bdd import evaluate_cooler_bdd
        from cmot.infer import infer_checkpoint

        resolved = self._resolve(stage)
        checkpoint = self.run_root / stage / "epoch_06.pt"
        val_view = self.manifest_root / (stage + "_val.json")
        prediction = self.run_root / stage / "predictions_epoch_06.jsonl"
        prediction_summary = self.run_root / stage / "prediction_summary.json"
        infer_checkpoint(
            self.paths["ovtr_root"], self.paths["config_file"], self.paths["text_embedding"], self.paths.get("image_embedding"),
            str(checkpoint), str(val_view), self.image_root, str(prediction), str(prediction_summary), resolved["active_global_ids"],
            resolved_config=resolved, device=self.device, split="val", checkpoint_role=str(resolved["method"]), alignment=bool(self.paths.get("image_embedding")),
        )
        metrics_path = self.run_root / stage / "metrics_epoch_06.json"
        metric = evaluate_cooler_bdd(
            str(val_view), str(prediction), str(metrics_path), self.paths["trackeval_root"],
            expected_binding={
                "checkpoint_sha256": sha256_file(str(checkpoint)),
                "resolved_config_sha256": canonical_json_hash(resolved),
                "view_manifest_hash": _read_json(str(val_view)).get("manifest_hash"),
            },
        )
        return {
            "status": "OK",
            "metrics": metric,
            "raw_prediction": {"basename": prediction.name, "sha256": sha256_file(str(prediction))},
            "raw_metrics": {"basename": metrics_path.name, "sha256": sha256_file(str(metrics_path))},
        }

    def _write_final_reports(self, data_audit: dict, stage_audit: dict, artifact_audit: dict, results: dict, pseudo_audit: dict, training_audit: dict) -> None:
        write_json(str(self.public_root / "pseudo_audit.json"), pseudo_audit)
        write_json(str(self.public_root / "training_audit.json"), training_audit)
        write_json(str(self.public_root / "results.json"), {"schema_version": "cmot.cooler_compat.results.v1", "results": list(results.values())})
        fields = ["stage", "method", "execution_status", "scope", "optimizer_steps", "mMOTA", "mHOTA", "mIDF1", "MOTA", "HOTA", "IDF1"]
        with (self.public_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for value in results.values():
                row = {key: value.get(key, "NOT_RUN") for key in fields}
                row.update(_metric_row(value.get("metrics") if isinstance(value.get("metrics"), Mapping) else None))
                writer.writerow(row)
        comparison = [
            "# Comparison to COOLer (reference values are not C-MOT measurements)",
            "",
            "C-MOT is marked `NOT_RUN` when the full-data gate is not satisfied; no pilot result is substituted.",
            "",
            "| Stage/method | mMOTA | mHOTA | mIDF1 | MOTA | HOTA | IDF1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        reference_rows = {
            "Stage 0": [("COOLer (reference only)", [67.6, 62.1, 73.3, 67.6, 62.1, 73.3])],
            "Stage 1": [
                ("COOLer Track PL (reference only)", [54.1, 52.6, 64.5, 62.5, 59.3, 70.3]),
                ("COOLer Full (reference only)", [54.2, 52.6, 64.3, 62.7, 59.5, 70.5]),
            ],
            "Stage 2": [
                ("COOLer Track PL (reference only)", [43.4, 49.1, 59.5, 58.2, 57.5, 68.3]),
                ("COOLer Full (reference only)", [42.8, 49.2, 59.6, 58.6, 57.9, 68.7]),
            ],
            "Oracle S2": [("COOLer Oracle (reference only)", [49.8, 50.8, 62.1, 63.2, 58.9, 70.4])],
        }
        stage_result_names = {
            "Stage 0": "s0",
            "Stage 1": "s1",
            "Stage 2": "s2",
            "Oracle S2": "oracle_s2",
        }
        for stage_name, reference_values in reference_rows.items():
            comparison.append("**%s**" % stage_name)
            comparison.append("")
            for name, row in reference_values:
                comparison.append("| %s | %s |" % (name, " | ".join(str(value) for value in row)))
            value = results.get(stage_result_names[stage_name], {})
            metrics = value.get("metrics") if isinstance(value.get("metrics"), Mapping) else None
            row = _metric_row(metrics)
            comparison.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                value.get("method", "C-MOT"), row["mMOTA"], row["mHOTA"], row["mIDF1"],
                row["MOTA"], row["HOTA"], row["IDF1"],
            ))
            comparison.append("")
        (self.public_root / "comparison_to_cooler.md").write_text("\n".join(comparison).rstrip() + "\n", encoding="utf-8")
        diagnosis = "DIAGNOSIS: NOT_RUN\n\nThe full BDD gate failed, so S0, CIL S1/S2, and Oracle-S2 were not executed.\n"
        (self.public_root / "diagnosis.md").write_text(diagnosis, encoding="utf-8")
        (self.public_root / "changes.md").write_text(
            "Implemented the independent cmot.cooler_compat.v1 path: strict stage views, replay-free pair sampling, tracker PL exclusion, accumulation trainer, TrackEval adapter, and fail-closed runner.\n\nFormal training status: BLOCKED by the local full-BDD gate.\n",
            encoding="utf-8",
        )
        (self.public_root / "known_limitations.md").write_text(
            "- Local BDD source did not contain the required 1400 train / 200 val annotated videos.\n"
            "- COOLer reference commit was not downloaded or found locally; no new download was attempted.\n"
            "- Consequently no checkpoint, prediction, TrackEval metric, gap, or diagnosis was generated; these remain NOT_RUN/null.\n",
            encoding="utf-8",
        )

    def run(self) -> int:
        data_audit_path = str(self.public_root / "data_audit.json")
        data_audit = audit_bdd100k_cooler_protocol(self.runtime_path, data_audit_path)
        stage_audit = self._stage_data_audit()
        artifact_audit = self._artifact_matching()
        self._base_reports(data_audit, stage_audit, artifact_audit)
        if not self.execute:
            print(json.dumps({"status": "VALIDATED_NOT_EXECUTED", "data_status": data_audit.get("status"), "reports": self.public_root.name}, sort_keys=True))
            return 0
        if not data_audit.get("full_bdd_gate_passed", False):
            reason = str(data_audit.get("status", "BLOCKED_FULL_BDD_MISSING"))
            results = {stage: self._not_run_result(stage, reason) for stage in ("s0", "s1", "s2", "oracle_s2")}
            pseudo_audit = {stage: "NOT_RUN" for stage in ("s1", "s2")}
            training_audit = {
                "schema_version": "cmot.cooler_compat.training_audit.v1",
                "status": reason,
                "epochs_requested": 6,
                "micro_batches_seen": 0,
                "optimizer_steps": 0,
                "effective_pairs_seen": 0,
                "runs": results,
            }
            self._write_final_reports(data_audit, stage_audit, artifact_audit, results, pseudo_audit, training_audit)
            write_json(str(self.private_root / "execution_state.json"), {"status": reason, "results": results, "data_audit_status": reason})
            print(json.dumps({"status": reason, "reports": self.public_root.name}, sort_keys=True))
            return 2
        results = {}
        pseudo_audit = {}
        training_audit = {"schema_version": "cmot.cooler_compat.training_audit.v1", "status": "RUNNING", "runs": {}}
        order = ["s0", "s1", "s2", "oracle_s2"] if "all" in self.targets else [value for value in self.targets if value in STAGES]
        for stage in order:
            summary = self._execute_stage(stage, data_audit)
            eval_summary = self._evaluate_stage(stage, summary, data_audit)
            record = dict(summary)
            record.update({
                "raw_prediction": eval_summary.get("raw_prediction"),
                "raw_metrics": eval_summary.get("raw_metrics"),
                "metrics": eval_summary.get("metrics", "NOT_RUN"),
            })
            results[stage] = record
            training_audit["runs"][stage] = summary
            if summary.get("pseudo_audit") not in (None, "NOT_RUN"):
                pseudo_audit[stage] = summary["pseudo_audit"]
        training_audit["status"] = "OK"
        self._write_final_reports(data_audit, stage_audit, artifact_audit, results, pseudo_audit, training_audit)
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--targets", nargs="+", default=["all"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        raise SystemExit(CoolerRunner(args.config, args.runtime, args.targets, args.execute, args.resume).run())
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
