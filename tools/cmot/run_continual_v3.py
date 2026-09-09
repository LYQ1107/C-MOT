"""Execute the V3 C-MOT DAG against an explicit private runtime binding.

The runner owns only orchestration.  It does not discover data, download
assets, or manufacture metrics: every training, inference, and TrackEval
result comes from the existing C-MOT call chain and is recorded with hashes.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ``python tools/cmot/run_continual_v3.py`` puts only tools/cmot on
# sys.path.  Make the checked-out project root explicit for the documented
# command without relying on the caller's PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cmot.config import load_config, resolve_runtime_config
from cmot.data.v3_view_builder import build_view, deterministic_split_ids, write_split_manifest
from cmot.evaluate import evaluate_trackeval
from cmot.infer import infer_checkpoint
from cmot.manifest import canonical_json_hash, sha256_file, write_json
from cmot.memory.clip_memory import ClipReplayMemory
from cmot.pseudo import calibrate_threshold, filter_prediction_records
from cmot.class_registry import tao_bdd_registry


CLASS_IDS = {"car": 206, "pedestrian": 792, "truck": 1122}
CLASS_NAMES = {value: key for key, value in CLASS_IDS.items()}
ALL_IDS = [206, 792, 1122]
_RUNTIME_CONTRACT_CACHE = {}


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash_without_field(value: Mapping[str, Any], field: str = "manifest_hash") -> str:
    data = dict(value)
    data.pop(field, None)
    return canonical_json_hash(data)


def _path_value(paths: Mapping[str, Any], key: str, required: bool = True) -> Optional[str]:
    value = paths.get(key)
    if value is None:
        if required:
            raise FileNotFoundError("runtime.paths.%s is not configured" % key)
        return None
    value = str(value)
    if required and not Path(value).exists():
        raise FileNotFoundError("runtime path %s does not exist: %s" % (key, value))
    return value


def _read_jsonl(path: str) -> Tuple[List[dict], dict]:
    records = []
    metadata = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("record_type") == "metadata":
                metadata.update(value.get("metadata", value))
            else:
                records.append(value)
    return records, metadata


def _write_jsonl(path: str, records: Iterable[Mapping[str, Any]], metadata: Optional[Mapping[str, Any]] = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        if metadata is not None:
            handle.write(json.dumps({"record_type": "metadata", "metadata": dict(metadata)}, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def _view_hash(path: str) -> str:
    payload = _read_json(path)
    declared = payload.get("manifest_hash")
    actual = _hash_without_field(payload)
    if declared and declared != actual:
        raise ValueError("view manifest hash mismatch: %s" % Path(path).name)
    return declared or actual


def _ensure_view(
    path: str,
    canonical: str,
    stage: str,
    mode: str,
    protocol_role: str,
    video_ids: Sequence[str],
    active_ids: Sequence[int],
    new_ids: Sequence[int],
    old_ids: Sequence[int],
    split: str,
    output_split: str,
    pl_path: Optional[str] = None,
    enable_pl: bool = False,
    total_pl_cap: Optional[int] = None,
    conflict_iou: float = 0.7,
    duplicate_iou: float = 0.7,
    min_segment_frames: int = 3,
    max_gap_s: float = 1.0,
) -> dict:
    destination = Path(path)
    if destination.is_file():
        payload = _read_json(path)
        if payload.get("stage") != stage or payload.get("protocol_role") != protocol_role:
            raise ValueError("existing view has a different protocol role/stage: %s" % destination.name)
        if [int(v) for v in payload.get("active_global_ids", [])] != [int(v) for v in active_ids]:
            raise ValueError("existing view active IDs mismatch: %s" % destination.name)
        stale_pl = payload.get("pl_source") != (None if not pl_path else Path(pl_path).name)
        if pl_path:
            try:
                _, pl_metadata = _read_jsonl(pl_path)
            except (OSError, ValueError, json.JSONDecodeError):
                pl_metadata = {}
            for key in (
                "pseudo_policy_version", "pseudo_policy_hash", "teacher_checkpoint_sha256",
                "raw_prediction_sha256", "calibration_view_sha256",
            ):
                if payload.get("pl_metadata", {}).get(key) != pl_metadata.get(key):
                    stale_pl = True
                    break
        if not stale_pl:
            _view_hash(path)
            return payload
        # A generated view is disposable.  Rebuild it against the current
        # validated segment manifest so a stale cache cannot silently retain
        # an older PL selection or reintroduce a frame-level cap.
    return build_view(
        canonical,
        path,
        stage,
        mode,
        protocol_role,
        video_ids,
        active_ids,
        new_ids,
        old_ids,
        pl_path=pl_path,
        conflict_iou=conflict_iou,
        duplicate_iou=duplicate_iou,
        split=split,
        output_split=output_split,
        enable_pl=enable_pl,
        total_pl_cap=total_pl_cap,
        min_segment_frames=min_segment_frames,
        max_gap_s=max_gap_s,
    )


def _ensure_split(path: str, canonical: str, stage: str, eval_ids: Sequence[str], config: Mapping[str, Any]) -> dict:
    if Path(path).is_file():
        payload = _read_json(path)
        if payload.get("source_manifest_sha256") != sha256_file(canonical):
            raise ValueError("existing split manifest source hash mismatch: %s" % Path(path).name)
        return payload
    protocol = dict(config["protocol"])
    payload = deterministic_split_ids(
        canonical,
        stage,
        int(protocol.get("train_source_cap", 32)),
        int(protocol.get("dev_source_cap", 8)),
        int(protocol.get("calibration_source_cap", 8)),
        eval_ids,
        int(config["seed"]),
    )
    return write_split_manifest(path, payload)


def _legal_gt_by_frame(view: Mapping[str, Any], allowed_ids: Optional[Sequence[int]] = None) -> Dict[str, List[dict]]:
    allowed = None if allowed_ids is None else {int(v) for v in allowed_ids}
    result = defaultdict(list)
    for video in view.get("videos", []):
        for frame in video.get("frames", []):
            for ann in frame.get("annotations", []):
                if ann.get("label_source", "gt") not in ("gt", "gt_replay"):
                    continue
                if ann.get("label_status") == "ignore" or int(ann.get("iscrowd", 0)):
                    continue
                gid = int(ann.get("global_semantic_id", -1))
                if allowed is None or gid in allowed:
                    result[str(frame["frame_key"])].append(ann)
    return dict(result)


def _safe_basename_ref(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    return {"basename": Path(path).name, "sha256": sha256_file(path)}


def _metric_scalars(metrics: Mapping[str, Any], global_ids: Sequence[int]) -> dict:
    rows = [metrics.get("classes", {}).get(str(int(gid))) for gid in global_ids]
    rows = [row for row in rows if row]
    if not rows:
        return {key: "NOT_RUN" for key in ("HOTA_mean", "IDF1", "MOTA", "DetA_mean", "AssA_mean")}
    return {
        key: sum(float(row[key]) for row in rows) / float(len(rows))
        for key in ("HOTA_mean", "IDF1", "MOTA", "DetA_mean", "AssA_mean")
    }


def _metric_recall(row: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not row or not isinstance(row.get("TP"), list) or not isinstance(row.get("FN"), list):
        return None
    if not row["TP"] or not row["FN"]:
        return None
    denom = float(row["TP"][0]) + float(row["FN"][0])
    return None if denom <= 0 else float(row["TP"][0]) / denom


def _result_record(
    stage: str,
    method: str,
    resolved: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    eval_summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    parent_checkpoint: Optional[str],
    teacher_checkpoint: Optional[str] = None,
    pl_path: Optional[str] = None,
    pl_audit: Optional[Mapping[str, Any]] = None,
    memory_version: Optional[str] = None,
    evaluation_scope: str = "pilot",
) -> dict:
    active_ids = [int(v) for v in resolved.get("active_global_ids", [])]
    old_ids = [int(v) for v in resolved.get("old_global_ids", [])]
    new_ids = [int(v) for v in resolved.get("new_global_ids", [])]
    all_scalars = _metric_scalars(metrics, active_ids)
    old_scalars = _metric_scalars(metrics, old_ids)
    new_scalars = _metric_scalars(metrics, new_ids)
    prediction_path = str(eval_summary.get("output_jsonl_path", ""))
    metric_path = str(eval_summary.get("metrics_path", ""))
    inference = dict(resolved.get("inference", {}))
    checkpoint_metadata = dict(train_summary.get("checkpoint_metadata", {}))
    actual_modules = dict(checkpoint_metadata.get("actual_modules", {}))
    actual_loss_weights = dict(checkpoint_metadata.get("actual_loss_weights", {}))
    if not actual_modules or not actual_loss_weights:
        # Backfill only metadata for runs created before V3 checkpoint audits;
        # this never changes the checkpoint or any metric artifact.
        try:
            contract_key = canonical_json_hash({
                "clip_frames": dict(resolved.get("training", {})).get("clip_frames", 4),
                "motion": dict(resolved.get("motion", {})).get("mode", "none"),
                "label_mode": resolved.get("label_mode", "complete"),
                "active_global_ids": list(resolved.get("active_global_ids", [])),
                "kd": bool(resolved.get("enable_kd", False)),
            })
            runtime_contract = _RUNTIME_CONTRACT_CACHE.get(contract_key)
            if runtime_contract is None:
                from cmot.ovtr_runtime import make_model
                runtime_paths = dict(resolved.get("runtime_paths", {}))
                audit_model, audit_criterion, _, _, _ = make_model(
                    runtime_paths["ovtr_root"], runtime_paths["config_file"],
                    runtime_paths["text_embedding"], runtime_paths["image_embedding"],
                    [int(value) for value in resolved.get("active_global_ids", [])], "cpu",
                    clip_len=int(dict(resolved.get("training", {})).get("clip_frames", 4)),
                    motion_mode=str(dict(resolved.get("motion", {})).get("mode", "none")),
                    label_mode=str(resolved.get("label_mode", "complete")),
                    alignment=True, resolved_config=resolved,
                )
                runtime_contract = {
                    "modules": {
                        "model_class": audit_model.__class__.__name__,
                        "criterion_class": audit_criterion.__class__.__name__,
                        "motion_head_class": None if getattr(audit_model, "motion_head", None) is None else audit_model.motion_head.__class__.__name__,
                        "motion_mode": str(dict(resolved.get("motion", {})).get("mode", "none")),
                        "kd_module_class": "ReplayAlignedDistillation" if bool(resolved.get("enable_kd", False)) else None,
                    },
                    "loss_weights": {str(name): float(value) for name, value in sorted(audit_criterion.weight_dict.items())},
                }
                if bool(resolved.get("enable_kd", False)):
                    runtime_contract["loss_weights"]["loss_kd"] = 1.0
                _RUNTIME_CONTRACT_CACHE[contract_key] = runtime_contract
                del audit_criterion
                del audit_model
        except Exception:
            runtime_contract = {}
        actual_modules = actual_modules or dict(runtime_contract.get("modules", {}))
        actual_loss_weights = actual_loss_weights or dict(runtime_contract.get("loss_weights", {}))
    return {
        "method": method,
        "stage": stage,
        "protocol_role": resolved.get("protocol_role"),
        "data_protocol": "diagnostic_joint" if resolved.get("protocol_role") == "diagnostic_joint" else "cil",
        "execution_status": "COMPLETE",
        "evaluation_scope": evaluation_scope,
        "active_global_ids": active_ids,
        "old_global_ids": old_ids,
        "new_global_ids": new_ids,
        "steps_requested": int(train_summary.get("steps_requested", 0)),
        "optimizer_steps": int(train_summary.get("steps_completed", 0)),
        "model_sha256": train_summary.get("checkpoint_sha256"),
        "checkpoint": {
            "basename": train_summary.get("checkpoint"),
            "sha256": train_summary.get("checkpoint_sha256"),
        },
        "parent_checkpoint_sha256": None if not parent_checkpoint else sha256_file(parent_checkpoint),
        "teacher_checkpoint_sha256": None if not teacher_checkpoint else sha256_file(teacher_checkpoint),
        "resolved_config_sha256": train_summary.get("checkpoint_metadata", {}).get("resolved_config_sha256"),
        "current_view_sha256": train_summary.get("train_view_sha256"),
        "replay_view_sha256": train_summary.get("replay_view_sha256"),
        "pl_manifest_sha256": None if not pl_path else sha256_file(pl_path),
        "memory_version": memory_version,
        "sampler_plan_hash": train_summary.get("sampler", {}).get("plan_hash"),
        "exposure": train_summary.get("exposure", {}),
        "independent_train_source_video_count": train_summary.get("exposure", {}).get("source_video_uids_unique"),
        "kd_replay_batches_seen": (
            train_summary.get("kd_replay_batches_seen", 0)
            if bool(resolved.get("enable_kd", False)) else "NOT_RUN"
        ),
        "kd_diagnostics": (
            dict(train_summary.get("kd_aggregate", {}))
            if bool(resolved.get("enable_kd", False)) else "NOT_RUN"
        ),
        "valid_kd_objects": (
            train_summary.get("kd_aggregate", {}).get("valid_kd_objects", 0.0)
            if bool(resolved.get("enable_kd", False)) else "NOT_RUN"
        ),
        "inference_thresholds": {
            key: inference.get(key)
            for key in ("birth_threshold", "keep_threshold", "export_threshold", "miss_tolerance", "duplicate_iou", "duplicate_feature_cos")
        },
        "modules": {
            **actual_modules,
            "motion_mode": resolved.get("motion", {}).get("mode", "none"),
            "pl_enabled": bool(resolved.get("enable_pl", False)),
            "kd_enabled": bool(resolved.get("enable_kd", False)),
            "inference_dedup_enabled": bool(inference.get("inference_dedup_enabled", True)),
        },
        "actual_modules": actual_modules,
        "actual_loss_weights": actual_loss_weights,
        "teacher_admission": None if pl_audit is None else {
            key: value for key, value in dict(pl_audit).items()
            if key not in ("pl_path",)
        },
        "raw_prediction": _safe_basename_ref(prediction_path),
        "raw_metrics": _safe_basename_ref(metric_path),
        "prediction_sha256": eval_summary.get("prediction_sha256"),
        "metrics": {
            "per_class": metrics.get("classes", {}),
            "old_macro": old_scalars,
            "new_macro": new_scalars,
            "all_seen_macro": all_scalars,
            "official_det_average": metrics.get("combined", {}),
            "combined_class_average": metrics.get("combined_class_average", {}),
        },
        "new_class_recall_at_hota_005": _metric_recall(metrics.get("classes", {}).get(str(new_ids[0]))) if new_ids else None,
        "fp_gt_ratio": (
            None if not metrics.get("combined", {}).get("gt_dets")
            else float(metrics["combined"].get("pred_dets", 0)) / float(metrics["combined"].get("gt_dets", 1))
        ),
    }


def _skipped_record(stage: str, method: str, resolved: Mapping[str, Any], reason: str, reference: Optional[Mapping[str, Any]] = None) -> dict:
    return {
        "method": method,
        "stage": stage,
        "protocol_role": resolved.get("protocol_role"),
        "execution_status": "SKIPPED_EQUIVALENT",
        "evaluation_scope": "pilot",
        "optimizer_steps": 0,
        "steps_requested": int(dict(resolved.get("training", {})).get("incremental_steps", 600)),
        "reason": reason,
        "reference_method": None if reference is None else reference.get("method"),
        "reference_result": None if reference is None else reference.get("raw_metrics"),
        "metrics": "NOT_RUN",
        "raw_prediction": "NOT_RUN",
        "raw_metrics": "NOT_RUN",
        "valid_kd_objects": "NOT_RUN",
        "teacher_admission": "NOT_RUN",
    }


def _not_run_record(stage: str, method: str, resolved: Mapping[str, Any], reason: str, status: str = "NOT_RUN") -> dict:
    training = dict(resolved.get("training", {}))
    return {
        "method": method,
        "stage": stage,
        "protocol_role": resolved.get("protocol_role"),
        "data_protocol": "cil",
        "execution_status": status,
        "evaluation_scope": "pilot",
        "active_global_ids": [int(value) for value in resolved.get("active_global_ids", [])],
        "old_global_ids": [int(value) for value in resolved.get("old_global_ids", [])],
        "new_global_ids": [int(value) for value in resolved.get("new_global_ids", [])],
        "optimizer_steps": 0,
        "steps_requested": int(training.get("incremental_steps", 600)),
        "reason": reason,
        "metrics": "NOT_RUN",
        "raw_prediction": "NOT_RUN",
        "raw_metrics": "NOT_RUN",
        "valid_kd_objects": "NOT_RUN",
        "teacher_admission": "NOT_RUN",
    }


def _blocked_kd_record(
    stage: str,
    method: str,
    resolved: Mapping[str, Any],
    parent_checkpoint: str,
    teacher_checkpoint: str,
    train_view: str,
    replay_view: str,
    failure,
) -> dict:
    return {
        "method": method,
        "stage": stage,
        "protocol_role": resolved.get("protocol_role"),
        "data_protocol": "cil",
        "execution_status": str(failure.status),
        "evaluation_scope": "pilot",
        "active_global_ids": [int(value) for value in resolved.get("active_global_ids", [])],
        "old_global_ids": [int(value) for value in resolved.get("old_global_ids", [])],
        "new_global_ids": [int(value) for value in resolved.get("new_global_ids", [])],
        "optimizer_steps": 0,
        "observed_optimizer_steps": max(0, int(failure.step) - 1),
        "steps_requested": int(dict(resolved.get("training", {})).get("incremental_steps", 600)),
        "reason": str(failure),
        "kd_fail_fast_step": int(failure.step),
        "kd_replay_batches_seen": int(failure.aggregate.get("kd_replay_batches_seen", 0)),
        "kd_diagnostics": {
            "last": dict(failure.diagnostics),
            "aggregate": dict(failure.aggregate),
        },
        "valid_kd_objects": float(failure.aggregate.get("valid_kd_objects", 0.0)),
        "model_sha256": None,
        "checkpoint": None,
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "current_view_sha256": _view_hash(train_view),
        "replay_view_sha256": _view_hash(replay_view),
        "resolved_config_sha256": canonical_json_hash(resolved),
        "metrics": "NOT_RUN",
        "raw_prediction": "NOT_RUN",
        "raw_metrics": "NOT_RUN",
        "exposure": {},
        "modules": {
            "motion_mode": dict(resolved.get("motion", {})).get("mode", "none"),
            "pl_enabled": bool(resolved.get("enable_pl", False)),
            "kd_enabled": True,
        },
        "teacher_admission": "NOT_RUN",
    }


def _mark_kd_equivalent(value: Mapping[str, Any], reference: Optional[Mapping[str, Any]], reason: str) -> dict:
    """Retain an observed redundant run without counting it as KD evidence."""
    record = dict(value)
    observed = {
        "execution_status": value.get("execution_status"),
        "optimizer_steps": value.get("optimizer_steps", 0),
        "model_sha256": value.get("model_sha256"),
        "checkpoint": value.get("checkpoint"),
        "raw_prediction": value.get("raw_prediction"),
        "raw_metrics": value.get("raw_metrics"),
        "prediction_sha256": value.get("prediction_sha256"),
        "metrics": value.get("metrics"),
        "exposure": value.get("exposure", {}),
        "valid_kd_objects": value.get("valid_kd_objects", 0.0),
    }
    record.update({
        "execution_status": "SKIPPED_EQUIVALENT",
        "reason": reason,
        "equivalent_to": None if reference is None else reference.get("method"),
        "reference_method": None if reference is None else reference.get("method"),
        "reference_result": None if reference is None else reference.get("raw_metrics"),
        "observed_execution_status": observed["execution_status"],
        "observed_optimizer_steps": observed["optimizer_steps"],
        "observed_result": observed,
        "metrics": "NOT_RUN",
        "raw_prediction": "NOT_RUN",
        "raw_metrics": "NOT_RUN",
        "optimizer_steps": 0,
        "valid_kd_objects": 0.0,
    })
    return record


class V3Runner:
    def __init__(self, config_path: str, runtime_path: str, targets: Sequence[str], resume: bool = False):
        self.config_path = str(config_path)
        self.runtime_path = str(runtime_path)
        self.config = load_config(config_path)
        if self.config.get("schema_version") != "cmot.continual_v3":
            raise ValueError("run_continual_v3 requires cmot.continual_v3")
        self.runtime = _read_json(runtime_path)
        if self.runtime.get("schema_version") not in ("cmot.continual_v3.runtime.v1", "cmot.runtime-local.v1"):
            raise ValueError("unsupported V3 runtime schema")
        self.paths = dict(self.runtime.get("paths", {}))
        self.targets = tuple(value for value in targets if value)
        allowed = {"diagnose", "s0", "joint", "s1", "s2", "p3", "motion"}
        if not set(self.targets) <= allowed:
            raise ValueError("unknown V3 target: %s" % sorted(set(self.targets) - allowed))
        self.resume = bool(resume)
        self.root = Path(_path_value(self.paths, "output_root"))
        previous_root = _path_value(self.paths, "previous_run_root", required=False)
        self.previous_root = Path(previous_root) if previous_root else None
        self.manifest_root = self.root / "manifests"
        self.memory_root = self.root / "memories"
        self.teacher_root = self.root / "teachers"
        self.run_root = self.root / "runs"
        self.diagnosis_root = self.root / "diagnosis"
        for directory in (self.manifest_root, self.memory_root, self.teacher_root, self.run_root, self.diagnosis_root):
            directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "execution_state.json"
        self.state = _read_json(str(self.state_path)) if self.resume and self.state_path.is_file() else {
            "schema_version": "cmot.continual_v3.execution.v1",
            "config_sha256": sha256_file(config_path),
            "runtime_basename": Path(runtime_path).name,
            "canonical_manifest_sha256": None,
            "results": {},
            "artifacts": {},
            "events": [],
        }
        self.canonical = _path_value(self.paths, "canonical_manifest")
        self.eval_slice = _path_value(self.paths, "immutable_eval_manifest")
        if self.state.get("config_sha256") != sha256_file(config_path):
            raise ValueError("resume config hash differs from execution state")
        self._validate_runtime()
        self.historical_results = {}
        historical_report = REPO_ROOT / "reports" / "v3" / "results.json"
        if historical_report.is_file():
            payload = _read_json(str(historical_report))
            self.historical_results = {
                str(value.get("method")): value
                for value in payload.get("results", [])
                if value.get("method")
            }

    def _validate_runtime(self) -> None:
        required = (
            "ovtr_root", "config_file", "text_embedding", "image_embedding", "foundation_checkpoint",
            "canonical_manifest", "immutable_eval_manifest", "media_index", "image_root", "trackeval_root",
            "output_root",
        )
        for key in required:
            _path_value(self.paths, key)
        if self.previous_root is not None and not self.previous_root.is_dir():
            raise FileNotFoundError("previous_run_root does not exist: %s" % self.previous_root)
        canonical_payload = _read_json(self.canonical)
        eval_payload = _read_json(self.eval_slice)
        if eval_payload.get("source_manifest_sha256") != sha256_file(self.canonical):
            raise ValueError("immutable eval slice is not bound to canonical manifest")
        self.eval_ids = [str(v) for v in eval_payload.get("video_ids", [])]
        if len(self.eval_ids) != int(self.config["protocol"].get("eval_video_cap", 16)):
            raise ValueError("immutable eval video count does not match V3 contract")
        self.state["canonical_manifest_sha256"] = sha256_file(self.canonical)
        self.state["immutable_eval_manifest_sha256"] = sha256_file(self.eval_slice)
        self.state["canonical_manifest_hash"] = canonical_payload.get("manifest_hash")

    def _event(self, event: str, **values: Any) -> None:
        record = {"time": round(time.time(), 3), "event": event}
        record.update(values)
        self.state.setdefault("events", []).append(record)
        write_json(str(self.state_path), self.state)

    def _save_state(self) -> None:
        write_json(str(self.state_path), self.state)

    def _record_result(self, key: str, value: Mapping[str, Any]) -> dict:
        self.state.setdefault("results", {})[key] = dict(value)
        self._save_state()
        return dict(value)

    def _result(self, key: str) -> Optional[dict]:
        value = self.state.get("results", {}).get(key)
        return None if value is None else dict(value)

    def _historical_result(self, method: str) -> Optional[dict]:
        value = self.historical_results.get(str(method))
        return None if value is None else dict(value)

    def _checkpoint_for_result(self, result: Mapping[str, Any]) -> str:
        """Resolve a checkpoint from the current or immutable historical root."""
        method = str(result.get("method", ""))
        steps = int(result.get("optimizer_steps", 0) or 0)
        expected = result.get("model_sha256")
        checkpoint = result.get("checkpoint")
        if isinstance(checkpoint, dict):
            expected = checkpoint.get("sha256") or expected
        roots = [self.root]
        if self.previous_root is not None:
            roots.append(self.previous_root)
        for root in roots:
            candidate = root / "runs" / method / ("checkpoint_%03d.pt" % steps)
            if not candidate.is_file():
                continue
            actual = sha256_file(str(candidate))
            if expected and actual != str(expected):
                raise ValueError("checkpoint SHA mismatch for %s: %s != %s" % (method, actual, expected))
            return str(candidate)
        raise FileNotFoundError("checkpoint for %s at %d steps was not found in audited roots" % (method, steps))

    def _resolved(self, method: str, stage: str) -> dict:
        return resolve_runtime_config(self.config, self.paths, method, stage)

    def _actual_runtime_contract(self, resolved: Mapping[str, Any]) -> dict:
        """Build the configured runtime once to audit actual classes/weights.

        Older completed runs predate the checkpoint metadata fields added by
        V3.  This CPU-only construction recovers the actual OVTR criterion
        contract for their report without loading data, weights, or running a
        training/inference step.
        """
        key = canonical_json_hash({
            "clip_frames": dict(resolved.get("training", {})).get("clip_frames", 4),
            "motion": dict(resolved.get("motion", {})).get("mode", "none"),
            "alignment": True,
            "label_mode": resolved.get("label_mode", "complete"),
            "active_global_ids": list(resolved.get("active_global_ids", [])),
            "kd": bool(resolved.get("enable_kd", False)),
        })
        cache = getattr(self, "_runtime_contract_cache", None)
        if cache is None:
            cache = self._runtime_contract_cache = {}
        if key in cache:
            return cache[key]
        from cmot.ovtr_runtime import make_model
        model, criterion, _, _, _ = make_model(
            _path_value(self.paths, "ovtr_root"), _path_value(self.paths, "config_file"),
            _path_value(self.paths, "text_embedding"), _path_value(self.paths, "image_embedding"),
            [int(value) for value in resolved.get("active_global_ids", [])], "cpu",
            clip_len=int(dict(resolved.get("training", {})).get("clip_frames", 4)),
            motion_mode=str(dict(resolved.get("motion", {})).get("mode", "none")),
            label_mode=str(resolved.get("label_mode", "complete")),
            alignment=True, resolved_config=resolved,
        )
        weights = {str(name): float(value) for name, value in sorted(criterion.weight_dict.items())}
        if bool(resolved.get("enable_kd", False)):
            weights["loss_kd"] = 1.0
        actual = {
            "model_class": model.__class__.__name__,
            "criterion_class": criterion.__class__.__name__,
            "motion_head_class": None if getattr(model, "motion_head", None) is None else model.motion_head.__class__.__name__,
            "motion_mode": str(dict(resolved.get("motion", {})).get("mode", "none")),
            "kd_module_class": "ReplayAlignedDistillation" if bool(resolved.get("enable_kd", False)) else None,
        }
        result = {"modules": actual, "loss_weights": weights}
        cache[key] = result
        del criterion
        del model
        return result

    def _partition(self, stage: str) -> dict:
        name = "split_%s.json" % stage
        payload = _ensure_split(str(self.manifest_root / name), self.canonical, stage, self.eval_ids, self.config)
        self.state.setdefault("artifacts", {})["split_" + stage] = {
            "basename": name,
            "sha256": sha256_file(str(self.manifest_root / name)),
            "manifest_hash": payload.get("manifest_hash"),
        }
        return payload

    def _make_view(self, name: str, stage: str, mode: str, role: str, source_ids: Sequence[str], active: Sequence[int], new: Sequence[int], old: Sequence[int], source_split: str, output_split: str, pl_path: Optional[str] = None, enable_pl: bool = False) -> Tuple[str, dict]:
        partition = self._partition(stage)
        path = self.manifest_root / name
        return str(path), _ensure_view(
            str(path), self.canonical, stage, mode, role, source_ids, active, new, old,
            split=source_split, output_split=output_split, pl_path=pl_path, enable_pl=enable_pl,
            total_pl_cap=int(dict(self.config.get("pseudo", {})).get("total_cap", 40)),
            conflict_iou=float(dict(self.config.get("supervision", {})).get("conflict_iou", 0.7)),
            duplicate_iou=float(dict(self.config.get("supervision", {})).get("pl_duplicate_iou", 0.7)),
            min_segment_frames=int(dict(self.config.get("pseudo", {})).get("min_segment_frames", 3)),
            max_gap_s=float(dict(self.config.get("pseudo", {})).get("max_gap_s", 1.0)),
        )

    def _prepare_views(self) -> dict:
        views = {}
        s0 = self._partition("S0")
        j3 = self._partition("J3")
        s1 = self._partition("S1")
        s2 = self._partition("S2")
        views["S0_current"], _ = self._make_view("S0_current_train.json", "S0", "train", "cil", s0["train_video_ids"], [206], [206], [], "train", "train")
        views["S0_dev"], _ = self._make_view("S0_dev.json", "S0", "eval", "cil", s0["dev_video_ids"], [206], [206], [], "train", "val")
        views["S0_calibration"], _ = self._make_view("S0_calibration.json", "S0", "eval", "cil", s0["calibration_video_ids"], [206], [206], [], "train", "calibration")
        views["S0_eval"], _ = self._make_view("S0_eval.json", "S0", "eval", "cil", self.eval_ids, [206], [206], [], "val", "val")
        views["J3_current"], _ = self._make_view("J3_current_train.json", "J3", "train", "diagnostic_joint", j3["train_video_ids"], ALL_IDS, ALL_IDS, [], "train", "train")
        views["J3_dev"], _ = self._make_view("J3_dev.json", "J3", "eval", "diagnostic_joint", j3["dev_video_ids"], ALL_IDS, ALL_IDS, [], "train", "val")
        views["J3_eval"], _ = self._make_view("J3_eval.json", "J3", "eval", "diagnostic_joint", self.eval_ids, ALL_IDS, ALL_IDS, [], "val", "val")
        views["S1_current"], _ = self._make_view("S1_current_train.json", "S1", "train", "cil", s1["train_video_ids"], [206, 792], [792], [206], "train", "train")
        views["S1_eval"], _ = self._make_view("S1_eval.json", "S1", "eval", "cil", self.eval_ids, [206, 792], [792], [206], "val", "val")
        views["S2_current"], _ = self._make_view("S2_current_train.json", "S2", "train", "cil", s2["train_video_ids"], ALL_IDS, [1122], [206, 792], "train", "train")
        views["S2_eval"], _ = self._make_view("S2_eval.json", "S2", "eval", "cil", self.eval_ids, ALL_IDS, [1122], [206, 792], "val", "val")
        # Calibration and teacher-source views are never fed to the optimizer.
        views["S1_calibration"], _ = self._make_view("S1_calibration.json", "S1", "eval", "cil", s1["calibration_video_ids"], [206], [206], [], "train", "calibration")
        views["S1_teacher_source"], _ = self._make_view("S1_teacher_source.json", "S1", "eval", "cil", s1["train_video_ids"], [206], [206], [], "train", "train")
        views["S2_calibration"], _ = self._make_view("S2_calibration.json", "S2", "eval", "cil", s2["calibration_video_ids"], [206, 792], [206, 792], [], "train", "calibration")
        views["S2_teacher_source"], _ = self._make_view("S2_teacher_source.json", "S2", "eval", "cil", s2["train_video_ids"], [206, 792], [206, 792], [], "train", "train")
        self.state["artifacts"]["views"] = {key: {"basename": Path(value).name, "sha256": sha256_file(value), "manifest_hash": _view_hash(value)} for key, value in views.items()}
        self._save_state()
        return views

    def _diagnose(self) -> dict:
        key = "asset_check_existing"
        existing = self._result(key)
        if self.resume and existing:
            return existing
        prediction = _path_value(self.paths, "existing_prediction", required=False)
        view = _path_value(self.paths, "existing_prediction_view", required=False)
        if not prediction or not view:
            result = {"method": "asset_check", "execution_status": "NOT_RUN", "evaluation_scope": "asset_diagnosis", "reason": "existing prediction/view not present in runtime"}
            return self._record_result(key, result)
        from tools.cmot.diagnose_predictions import diagnose_predictions
        output = self.diagnosis_root / "existing_asset_check.json"
        result = diagnose_predictions(
            prediction, view, str(output),
            max_videos=int(dict(self.config.get("diagnosis", {})).get("max_videos", 4)),
            max_frames_per_video=int(dict(self.config.get("diagnosis", {})).get("max_frames_per_video", 80)),
            duplicate_iou=float(dict(self.config.get("inference", {})).get("duplicate_iou", 0.85)),
        )
        result["raw_prediction"] = _safe_basename_ref(prediction)
        result["diagnosis_file"] = _safe_basename_ref(str(output))
        return self._record_result(key, result)

    def _call_train(self, resolved: Mapping[str, Any], train_view: str, output_name: str, parent_checkpoint: str, init_mode: str, steps: int, replay_view: Optional[str] = None, teacher_checkpoint: Optional[str] = None, teacher_active: Sequence[int] = (206,), teacher_resolved: Optional[Mapping[str, Any]] = None, resume_checkpoint: Optional[str] = None) -> dict:
        from cmot.train import train
        output_dir = self.run_root / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return train(
            _path_value(self.paths, "ovtr_root"),
            _path_value(self.paths, "config_file"),
            _path_value(self.paths, "text_embedding"),
            _path_value(self.paths, "image_embedding"),
            train_view,
            _path_value(self.paths, "image_root"),
            str(output_dir),
            [int(v) for v in resolved["active_global_ids"]],
            int(steps),
            parent_checkpoint,
            str(self.runtime.get("device", "cuda")),
            str(resolved.get("motion", {}).get("mode", "none")),
            str(resolved.get("label_mode", "complete")),
            int(dict(resolved.get("training", {})).get("clip_frames", 4)),
            tuple(dict(resolved.get("training", {})).get("input_size", [640, 360])),
            100000,
            resume_checkpoint,
            True,
            int(resolved.get("seed", self.config["seed"])),
            resolved_config=resolved,
            replay_view=replay_view,
            experiment_id=output_name,
            init_mode=init_mode,
            replay_memory_version=resolved.get("replay_memory_version"),
            protocol_hash=resolved.get("protocol_hash"),
            teacher_checkpoint=teacher_checkpoint,
            teacher_active_global_ids=teacher_active,
            teacher_resolved_config=teacher_resolved,
        )

    def _checkpoint_path(self, output_name: str, steps: int) -> str:
        path = self.run_root / output_name / ("checkpoint_%03d.pt" % int(steps))
        if not path.is_file():
            raise FileNotFoundError("missing checkpoint %s" % path.name)
        return str(path)

    def _run_dev_eval(self, resolved: Mapping[str, Any], checkpoint: str, dev_view: str, output_name: str) -> dict:
        output_dir = self.run_root / output_name
        prediction = output_dir / "dev_step600.jsonl"
        prediction_summary = output_dir / "dev_step600_summary.json"
        metric_path = output_dir / "dev_step600_trackeval.json"
        if self.resume and metric_path.is_file() and prediction_summary.is_file():
            return _read_json(str(metric_path))
        infer_summary = infer_checkpoint(
            _path_value(self.paths, "ovtr_root"), _path_value(self.paths, "config_file"),
            _path_value(self.paths, "text_embedding"), _path_value(self.paths, "image_embedding"),
            checkpoint, dev_view, _path_value(self.paths, "image_root"), str(prediction), str(prediction_summary),
            resolved["active_global_ids"], resolved_config=resolved, device=str(self.runtime.get("device", "cuda")),
            split="val", alignment=True, checkpoint_role="dev_step600",
        )
        infer_summary["output_jsonl_path"] = str(prediction)
        metrics = evaluate_trackeval(
            dev_view, str(prediction), str(metric_path), _path_value(self.paths, "trackeval_root"),
            expected_binding={
                "checkpoint_sha256": sha256_file(checkpoint),
                "resolved_config_sha256": canonical_json_hash(resolved),
                "view_manifest_hash": _view_hash(dev_view),
            },
        )
        metrics["metrics_path"] = str(metric_path)
        return metrics

    def _train_long_stage(self, resolved: Mapping[str, Any], train_view: str, dev_view: str, output_name: str, parent_checkpoint: str, replay_view: Optional[str] = None, teacher_checkpoint: Optional[str] = None, teacher_active: Sequence[int] = (206,), teacher_resolved: Optional[Mapping[str, Any]] = None) -> dict:
        total = int(dict(resolved.get("training", {})).get("s0_steps", 1200) if resolved.get("stage") in ("S0", "J3") else dict(resolved.get("training", {})).get("incremental_steps", 600))
        if resolved.get("stage") == "J3":
            total = int(dict(resolved.get("training", {})).get("joint_steps", 1200))
        output_dir = self.run_root / output_name
        final_checkpoint = output_dir / ("checkpoint_%03d.pt" % total)
        summary_path = output_dir / "train_summary.json"
        if self.resume and final_checkpoint.is_file() and summary_path.is_file():
            return _read_json(str(summary_path))
        if total > 600:
            checkpoint600 = output_dir / "checkpoint_600.pt"
            if not checkpoint600.is_file():
                self._event("train_start", run=output_name, steps=600)
                self._call_train(resolved, train_view, output_name, parent_checkpoint, "foundation", 600, replay_view, teacher_checkpoint, teacher_active, teacher_resolved)
            dev_metric = output_dir / "dev_step600_trackeval.json"
            if not dev_metric.is_file():
                self._event("dev_eval", run=output_name, step=600)
                self._run_dev_eval(resolved, str(checkpoint600), dev_view, output_name)
            self._event("train_resume", run=output_name, from_step=600, to_step=total)
            self._call_train(resolved, train_view, output_name, parent_checkpoint, "resume", total, replay_view, teacher_checkpoint, teacher_active, teacher_resolved, resume_checkpoint=str(checkpoint600))
        else:
            self._event("train_start", run=output_name, steps=total)
            self._call_train(resolved, train_view, output_name, parent_checkpoint, "foundation", total, replay_view, teacher_checkpoint, teacher_active, teacher_resolved)
        if not final_checkpoint.is_file() or not summary_path.is_file():
            raise RuntimeError("training did not produce a proved final artifact: %s" % output_name)
        return _read_json(str(summary_path))

    def _train_stage(self, resolved: Mapping[str, Any], train_view: str, output_name: str, parent_checkpoint: str, replay_view: Optional[str] = None, teacher_checkpoint: Optional[str] = None, teacher_active: Sequence[int] = (206,), teacher_resolved: Optional[Mapping[str, Any]] = None) -> dict:
        total = int(dict(resolved.get("training", {})).get("incremental_steps", 600))
        output_dir = self.run_root / output_name
        final_checkpoint = output_dir / ("checkpoint_%03d.pt" % total)
        summary_path = output_dir / "train_summary.json"
        if self.resume and final_checkpoint.is_file() and summary_path.is_file():
            return _read_json(str(summary_path))
        self._event("train_start", run=output_name, steps=total)
        result = self._call_train(resolved, train_view, output_name, parent_checkpoint, "stage_transfer", total, replay_view, teacher_checkpoint, teacher_active, teacher_resolved)
        if not final_checkpoint.is_file():
            raise RuntimeError("training did not produce final checkpoint: %s" % output_name)
        return result

    def _evaluate(self, resolved: Mapping[str, Any], train_summary: Mapping[str, Any], eval_view: str, output_name: str) -> Tuple[dict, dict]:
        output_dir = self.run_root / output_name
        checkpoint = self._checkpoint_path(output_name, int(train_summary["steps_completed"]))
        prediction = output_dir / "eval.jsonl"
        prediction_summary = output_dir / "eval_summary.json"
        metric_path = output_dir / "trackeval.json"
        if self.resume and metric_path.is_file() and prediction_summary.is_file():
            infer_summary = _read_json(str(prediction_summary))
            metrics = _read_json(str(metric_path))
        else:
            infer_summary = infer_checkpoint(
                _path_value(self.paths, "ovtr_root"), _path_value(self.paths, "config_file"),
                _path_value(self.paths, "text_embedding"), _path_value(self.paths, "image_embedding"),
                checkpoint, eval_view, _path_value(self.paths, "image_root"), str(prediction), str(prediction_summary),
                resolved["active_global_ids"], resolved_config=resolved, device=str(self.runtime.get("device", "cuda")),
                split="val", alignment=True, checkpoint_role=output_name,
            )
            metrics = evaluate_trackeval(
                eval_view, str(prediction), str(metric_path), _path_value(self.paths, "trackeval_root"),
                expected_binding={
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "resolved_config_sha256": canonical_json_hash(resolved),
                    "view_manifest_hash": _view_hash(eval_view),
                },
            )
            write_json(str(metric_path), metrics)
        infer_summary["output_jsonl_path"] = str(prediction)
        infer_summary["metrics_path"] = str(metric_path)
        return infer_summary, metrics

    def _zero_vocab_eval(
        self,
        stage: str,
        parent_method: str,
        checkpoint: str,
        base_resolved: Mapping[str, Any],
        eval_view: str,
        vocabulary_role: str,
        active_ids: Sequence[int],
        previous_ids: Sequence[int],
    ) -> dict:
        """Evaluate one checkpoint with an explicitly overridden vocabulary.

        This is intentionally an inference-only artifact.  It uses the same
        checkpoint and frames for the previous and expanded vocabulary rows;
        no future labels enter the model and no optimizer state is touched.
        """
        key = "zero_%s_%s_%s_vocabulary" % (
            stage,
            parent_method,
            "previous" if vocabulary_role == "previous" else "expanded",
        )
        existing = self._result(key)
        if self.resume and existing and existing.get("execution_status") == "COMPLETE":
            return existing
        output_name = key.replace("-", "_")
        output_dir = self.run_root / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction = output_dir / "eval.jsonl"
        prediction_summary = output_dir / "eval_summary.json"
        metric_path = output_dir / "trackeval.json"
        resolved = dict(base_resolved)
        override = {
            "stage": stage,
            "parent_method": parent_method,
            "vocabulary_role": vocabulary_role,
            "checkpoint_active_global_ids": list(base_resolved.get("active_global_ids", [])),
            "evaluation_active_global_ids": [int(value) for value in active_ids],
            "optimizer_steps": 0,
        }
        resolved["evaluation_vocabulary_override"] = override
        checkpoint_sha = sha256_file(checkpoint)
        if not (self.resume and metric_path.is_file() and prediction_summary.is_file()):
            infer_summary = infer_checkpoint(
                _path_value(self.paths, "ovtr_root"), _path_value(self.paths, "config_file"),
                _path_value(self.paths, "text_embedding"), _path_value(self.paths, "image_embedding"),
                checkpoint, eval_view, _path_value(self.paths, "image_root"),
                str(prediction), str(prediction_summary), active_ids,
                resolved_config=resolved, device=str(self.runtime.get("device", "cuda")),
                split="val", alignment=True, checkpoint_role=key,
                allow_vocab_expansion=True,
                evaluation_vocabulary_override=override,
            )
            metrics = evaluate_trackeval(
                eval_view, str(prediction), str(metric_path), _path_value(self.paths, "trackeval_root"),
                expected_binding={
                    "checkpoint_sha256": checkpoint_sha,
                    "resolved_config_sha256": canonical_json_hash(resolved),
                    "view_manifest_hash": _view_hash(eval_view),
                },
            )
            infer_summary["output_jsonl_path"] = str(prediction)
            infer_summary["metrics_path"] = str(metric_path)
            write_json(str(prediction_summary), infer_summary)
        else:
            infer_summary = _read_json(str(prediction_summary))
            metrics = _read_json(str(metric_path))
        old_ids = [int(value) for value in previous_ids if int(value) in set(int(v) for v in active_ids)]
        new_ids = [int(value) for value in active_ids if int(value) not in set(old_ids)]
        record = {
            "method": key,
            "stage": stage,
            "protocol_role": "cil",
            "data_protocol": "cil",
            "execution_status": "COMPLETE",
            "evaluation_scope": "pilot",
            "zero_step": True,
            "optimizer_steps": 0,
            "steps_requested": 0,
            "vocabulary_role": vocabulary_role,
            "evaluation_vocabulary_override": override,
            "model_sha256": checkpoint_sha,
            "parent_checkpoint_sha256": checkpoint_sha,
            "teacher_checkpoint_sha256": None,
            "resolved_config_sha256": canonical_json_hash(resolved),
            "evaluation_view_sha256": sha256_file(eval_view),
            "sampler_plan_hash": None,
            "raw_prediction": _safe_basename_ref(str(prediction)),
            "raw_metrics": _safe_basename_ref(str(metric_path)),
            "prediction_sha256": infer_summary.get("prediction_sha256"),
            "metrics": {
                "per_class": metrics.get("classes", {}),
                "old_macro": _metric_scalars(metrics, old_ids),
                "new_macro": _metric_scalars(metrics, new_ids),
                "all_seen_macro": _metric_scalars(metrics, active_ids),
                "official_det_average": metrics.get("combined", {}),
                "combined_class_average": metrics.get("combined_class_average", {}),
            },
            "inference_thresholds": dict(base_resolved.get("inference", {})),
            "modules": {
                "motion_mode": base_resolved.get("motion", {}).get("mode", "none"),
                "pl_enabled": False,
                "kd_enabled": False,
                "inference_dedup_enabled": bool(dict(base_resolved.get("inference", {})).get("inference_dedup_enabled", True)),
            },
            "old_global_ids": old_ids,
            "new_global_ids": new_ids,
            "active_global_ids": [int(value) for value in active_ids],
            "valid_kd_objects": "NOT_RUN",
            "exposure": {
                "optimizer_steps": 0,
                "frames_evaluated": infer_summary.get("frames", "NOT_RUN"),
                "videos_evaluated": infer_summary.get("videos", "NOT_RUN"),
            },
        }
        record["old_class_recall_at_hota_005"] = None
        self._record_result(key, record)
        self.state.setdefault("artifacts", {}).setdefault("zero_vocab", {})[key] = {
            "prediction": _safe_basename_ref(str(prediction)),
            "metrics": _safe_basename_ref(str(metric_path)),
        }
        self._save_state()
        return record

    def _make_zero_vocab_view(self, stage: str, name: str, active_ids: Sequence[int]) -> str:
        path, _ = self._make_view(
            name, stage, "eval", "cil", self.eval_ids, active_ids, [], active_ids,
            "val", "val",
        )
        return path

    def _run_s1_zero_vocab(self, views: Mapping[str, str], checkpoint: str) -> None:
        old_view = self._make_zero_vocab_view("S1", "S1_zero_previous_eval.json", [206])
        base = self._resolved("S0-v3", "S0")
        self._zero_vocab_eval("S1", "S0-v3", checkpoint, base, old_view, "previous", [206], [206])
        self._zero_vocab_eval("S1", "S0-v3", checkpoint, base, views["S1_eval"], "expanded", [206, 792], [206])

    def _run_s2_zero_vocab(self, views: Mapping[str, str], s1_results: Mapping[str, Mapping[str, Any]]) -> None:
        specs = (
            ("R-QPLSEG-S1", [206, 792], "NOT_RUN_PARENT_QPLSEG_INVALID"),
            ("R-QPLSEG-KD-S1", [206, 792], "NOT_RUN_PARENT_KD_INVALID"),
        )
        old_view = self._make_zero_vocab_view("S2", "S2_zero_previous_eval.json", [206, 792])
        for parent_method, previous_ids, missing_status in specs:
            parent = s1_results.get(parent_method) or self._result(parent_method)
            if not parent or parent.get("execution_status") != "COMPLETE":
                for role in ("previous", "expanded"):
                    key = "zero_S2_%s_%s_vocabulary" % (parent_method, role)
                    base = self._resolved(parent_method.rsplit("-S1", 1)[0], "S1")
                    record = _not_run_record(
                        "S2", key, base,
                        "parent %s checkpoint was not complete" % parent_method,
                        missing_status,
                    )
                    record.update({"zero_step": True, "vocabulary_role": role, "steps_requested": 0})
                    self._record_result(key, record)
                continue
            checkpoint = self._checkpoint_for_result(parent)
            base = self._resolved(parent_method.rsplit("-S1", 1)[0], "S1")
            self._zero_vocab_eval("S2", parent_method, checkpoint, base, old_view, "previous", previous_ids, previous_ids)
            self._zero_vocab_eval("S2", parent_method, checkpoint, base, views["S2_eval"], "expanded", [206, 792, 1122], previous_ids)

    def _build_memory(self, stage: str, source_view: str, active_ids: Sequence[int], name: str, old_view_name: str) -> Tuple[str, dict, str]:
        protocol_hash = canonical_json_hash({
            "schema": "cmot.continual_v3",
            "canonical_manifest_sha256": sha256_file(self.canonical),
            "eval_manifest_sha256": sha256_file(self.eval_slice),
            "seed": int(self.config["seed"]),
        })
        registry_hash = canonical_json_hash(tao_bdd_registry().as_dict())
        memory_path = self.memory_root / name
        if memory_path.is_file():
            memory = ClipReplayMemory.load(str(memory_path), protocol_hash, registry_hash)
        else:
            historical_path = None
            if self.previous_root is not None:
                candidate = self.previous_root / "memories" / name
                if candidate.is_file():
                    historical_path = candidate
            if historical_path is None:
                raise FileNotFoundError(
                    "legal replay snapshot %s is absent; refusing to rebuild memory from canonical data" % name
                )
            memory = ClipReplayMemory.load(str(historical_path), protocol_hash, registry_hash)
            memory.save(str(memory_path))
        replay_payload = memory.as_view(stage, active_ids, split="train")
        replay_path = self.memory_root / old_view_name
        if not replay_path.is_file() or not self.resume:
            write_json(str(replay_path), replay_payload)
        return str(memory_path), _read_json(str(memory_path)), str(replay_path)

    def _pseudo_policy_hash(self, stage: str, old_ids: Sequence[int]) -> str:
        pseudo = dict(self.config.get("pseudo", {}))
        supervision = dict(self.config.get("supervision", {}))
        calibration = dict(self.config.get("calibration", {}))
        return canonical_json_hash({
            "stage": str(stage),
            "old_global_ids": [int(value) for value in old_ids],
            "selector_implementation": "qpl_segment_v2_cap_fix_1",
            "calibration": calibration,
            "pseudo": pseudo,
            "conflict_iou": float(supervision.get("conflict_iou", 0.7)),
            "pl_duplicate_iou": float(supervision.get("pl_duplicate_iou", 0.7)),
        })

    def _teacher_prediction_source(
        self,
        stage: str,
        suffix: str,
        teacher_checkpoint: str,
        view: str,
        active_ids: Sequence[int],
        split: str,
        teacher_resolved: Mapping[str, Any],
    ) -> Tuple[str, dict]:
        """Reuse a matching raw prediction, otherwise infer into this run root."""
        expected_hash = sha256_file(teacher_checkpoint)
        filename = "%s_%s.jsonl" % (stage, suffix)
        summary_name = filename + ".summary.json"
        current_path = self.teacher_root / filename
        candidates = [current_path]
        if self.previous_root is not None:
            candidates.append(self.previous_root / "teachers" / filename)
        for candidate in candidates:
            if not candidate.is_file():
                continue
            _, metadata = _read_jsonl(str(candidate))
            if metadata.get("teacher_checkpoint_sha256") == expected_hash:
                # Historical V3 used both ``foo.jsonl.summary.json`` and
                # ``foo.summary.json``.  The raw metadata is the binding
                # authority; tolerate either summary spelling when reusing a
                # hash-matched raw prediction.
                return str(candidate), metadata
        current_path.parent.mkdir(parents=True, exist_ok=True)
        infer_checkpoint(
            _path_value(self.paths, "ovtr_root"), _path_value(self.paths, "config_file"),
            _path_value(self.paths, "text_embedding"), _path_value(self.paths, "image_embedding"),
            teacher_checkpoint, view, _path_value(self.paths, "image_root"), str(current_path),
            str(current_path.parent / summary_name), active_ids,
            resolved_config=teacher_resolved, device=str(self.runtime.get("device", "cuda")),
            split=split, alignment=True, checkpoint_role="teacher_%s" % suffix,
        )
        _, metadata = _read_jsonl(str(current_path))
        if metadata.get("teacher_checkpoint_sha256") != expected_hash:
            raise ValueError("teacher raw prediction hash does not match checkpoint for %s" % stage)
        return str(current_path), metadata

    def _prepare_teacher_pl(
        self,
        stage: str,
        teacher_checkpoint: str,
        teacher_resolved: Mapping[str, Any],
        teacher_source_view: str,
        calibration_view: str,
        current_view: str,
        old_ids: Sequence[int],
        current_new_ids: Sequence[int],
    ) -> Tuple[Optional[str], dict]:
        pseudo = dict(self.config.get("pseudo", {}))
        policy_version = str(pseudo.get("policy_version", "qpl_segment_v2"))
        if policy_version != "qpl_segment_v2":
            raise ValueError("V3.1 QPLSEG requires pseudo.policy_version=qpl_segment_v2")
        teacher_hash = sha256_file(teacher_checkpoint)
        raw_path, _ = self._teacher_prediction_source(
            stage, "teacher_raw", teacher_checkpoint, teacher_source_view, old_ids, "train", teacher_resolved
        )
        calibration_raw, _ = self._teacher_prediction_source(
            stage, "calibration_raw", teacher_checkpoint, calibration_view, old_ids, "calibration", teacher_resolved
        )
        raw_records, _ = _read_jsonl(raw_path)
        calibration_records, _ = _read_jsonl(calibration_raw)
        calibration_gt = _legal_gt_by_frame(_read_json(calibration_view), old_ids)
        calibration_cfg = dict(self.config.get("calibration", {}))
        thresholds = {}
        per_class_calibration = {}
        disabled = []
        for global_id in old_ids:
            value = calibrate_threshold(
                calibration_records,
                calibration_gt,
                thresholds=calibration_cfg.get("thresholds", [0.5, 0.6, 0.7, 0.8, 0.9]),
                iou_min=float(calibration_cfg.get("iou_min", 0.5)),
                min_predictions=int(calibration_cfg.get("min_predictions", 30)),
                min_videos=int(calibration_cfg.get("min_videos", 3)),
                wilson_lower_min=float(calibration_cfg.get("wilson_lower_min", 0.8)),
                old_global_ids=[int(global_id)],
            )
            per_class_calibration[str(global_id)] = value
            selected = value.get("selected")
            if selected is None:
                disabled.append(int(global_id))
            else:
                thresholds[int(global_id)] = float(selected["threshold"])

        configured_caps = dict(pseudo.get("per_class_segment_cap", {}))
        per_class_segment_caps = {
            int(global_id): int(configured_caps.get(global_id, configured_caps.get(str(global_id), pseudo.get("total_segment_cap", 40))))
            for global_id in old_ids
        }
        for global_id in old_ids:
            if per_class_segment_caps.get(int(global_id), 0) <= 0:
                disabled.append(int(global_id))
                thresholds.pop(int(global_id), None)
        disabled = sorted(set(disabled))
        conflict = _legal_gt_by_frame(_read_json(current_view), current_new_ids)
        accepted: List[dict] = []
        filter_audit: dict = {}
        if thresholds:
            accepted, filter_audit = filter_prediction_records(
                raw_records,
                teacher_hash,
                old_ids,
                thresholds,
                {int(key): int(value) for key, value in per_class_segment_caps.items() if int(key) in thresholds},
                total_segment_cap=int(pseudo.get("total_segment_cap", 40)),
                total_frame_cap=int(pseudo.get("total_frame_cap", 256)),
                max_segment_frames=int(pseudo.get("max_segment_frames", 8)),
                min_segment_frames=int(pseudo.get("min_segment_frames", 3)),
                max_gap_s=float(pseudo.get("max_gap_s", 1.0)),
                conflict_gt_by_frame=conflict,
                conflict_iou=float(dict(self.config.get("supervision", {})).get("conflict_iou", 0.7)),
                duplicate_iou=float(dict(self.config.get("supervision", {})).get("pl_duplicate_iou", 0.7)),
            )
        raw_by_frame = {str(record.get("frame_key")): record for record in raw_records}
        accepted_by_frame = defaultdict(list)
        for value in accepted:
            accepted_by_frame[str(value["frame_key"])].append({
                "track_id": int(value["track_id"]),
                "teacher_track_id": int(value["teacher_track_id"]),
                "global_id": int(value["global_id"]),
                "score": float(value["score"]),
                "bbox_xyxy": list(value["bbox_xyxy"]),
                "pl_segment_id": str(value["pl_segment_id"]),
                "segment_length": int(value["segment_length"]),
                "segment_reliability": float(value["segment_reliability"]),
                "segment_mean_score": float(value["segment_mean_score"]),
                "reliability": float(value["reliability"]),
                "teacher_checkpoint_sha256": teacher_hash,
            })
        policy_hash = self._pseudo_policy_hash(stage, old_ids)
        pl_path = self.teacher_root / ("%s_qpl_segment_v2.jsonl" % stage)
        pl_metadata = {
            "schema_version": "cmot.continual_v3.pl",
            "pseudo_policy_version": policy_version,
            "pseudo_policy_hash": policy_hash,
            "teacher_checkpoint_sha256": teacher_hash,
            "raw_prediction_sha256": sha256_file(raw_path),
            "calibration_view_sha256": sha256_file(calibration_view),
            "calibration": per_class_calibration,
            "thresholds": {str(key): value for key, value in sorted(thresholds.items())},
            "per_class_segment_cap": {str(key): value for key, value in sorted(per_class_segment_caps.items())},
            "total_segment_cap": int(pseudo.get("total_segment_cap", 40)),
            "total_frame_cap": int(pseudo.get("total_frame_cap", 256)),
            "disabled_global_ids": disabled,
            "filter_audit": filter_audit,
        }
        cached_metadata = {}
        if pl_path.is_file():
            try:
                _, cached_metadata = _read_jsonl(str(pl_path))
            except (OSError, ValueError, json.JSONDecodeError):
                cached_metadata = {}
        required_metadata = {
            "pseudo_policy_version": policy_version,
            "pseudo_policy_hash": policy_hash,
            "teacher_checkpoint_sha256": teacher_hash,
            "raw_prediction_sha256": sha256_file(raw_path),
            "calibration_view_sha256": sha256_file(calibration_view),
        }
        if cached_metadata and all(cached_metadata.get(key) == value for key, value in required_metadata.items()):
            pass
        else:
            records = []
            for frame_key, predictions in sorted(accepted_by_frame.items()):
                raw_record = raw_by_frame.get(frame_key, {})
                records.append({
                    "frame_key": frame_key,
                    "video_uid": str(raw_record.get("video_uid") or frame_key.rsplit("/", 1)[0]),
                    "frame_index": raw_record.get("frame_index"),
                    "timestamp_s": raw_record.get("timestamp_s"),
                    "predictions": predictions,
                })
            _write_jsonl(str(pl_path), records, pl_metadata)
        audit = {
            "status": "OK" if accepted else ("PL_DISABLED_LOW_SUPPORT_OR_QUALITY" if disabled else "PL_EMPTY_AFTER_SEGMENT_GATES"),
            "pseudo_policy_version": policy_version,
            "pseudo_policy_hash": policy_hash,
            "teacher_checkpoint_sha256": teacher_hash,
            "raw_prediction": _safe_basename_ref(raw_path),
            "calibration_prediction": _safe_basename_ref(calibration_raw),
            "calibration_view_sha256": sha256_file(calibration_view),
            "calibration": per_class_calibration,
            "thresholds": {str(key): value for key, value in sorted(thresholds.items())},
            "per_class_segment_cap": {str(key): value for key, value in sorted(per_class_segment_caps.items())},
            "disabled_global_ids": disabled,
            "filter": filter_audit,
            "candidate_segments": int(filter_audit.get("candidate_segments", 0)),
            "selected_segments": int(filter_audit.get("selected_segments", 0)),
            "selected_frame_predictions": int(filter_audit.get("selected_frame_predictions", 0)),
            "selected_segments_by_class": filter_audit.get("selected_segments_by_class", {}),
            "selected_frames_by_class": filter_audit.get("selected_frames_by_class", {}),
            "selected_source_videos_by_class": filter_audit.get("selected_source_videos_by_class", {}),
            "rejected_short_segments": int(filter_audit.get("rejected_short_segments", 0)),
            "rejected_gt_conflict_segments": int(filter_audit.get("rejected_gt_conflict_segments", 0)),
            "rejected_duplicate_segments": int(filter_audit.get("rejected_duplicate_segments", 0)),
            "rejected_segment_budget": int(filter_audit.get("rejected_segment_budget", 0)),
            "rejected_frame_budget": int(filter_audit.get("rejected_frame_budget", 0)),
            "pl_path": str(pl_path),
        }
        write_json(str(self.teacher_root / ("%s_admission.json" % stage)), audit)
        if not accepted:
            return None, audit
        return str(pl_path), audit

    def _run_s0(self, views: Mapping[str, str]) -> Optional[dict]:
        key = "S0-v3"
        if self.resume and self._result(key):
            return self._result(key)
        resolved = self._resolved("S0-v3", "S0")
        resolved["protocol_hash"] = canonical_json_hash({"canonical": sha256_file(self.canonical), "eval": sha256_file(self.eval_slice), "seed": self.config["seed"]})
        train_summary = self._train_long_stage(resolved, views["S0_current"], views["S0_dev"], "S0-v3", _path_value(self.paths, "foundation_checkpoint"))
        checkpoint = self._checkpoint_path("S0-v3", int(train_summary["steps_completed"]))
        eval_summary, metrics = self._evaluate(resolved, train_summary, views["S0_eval"], "S0-v3")
        record = _result_record("S0", "S0-v3", resolved, train_summary, eval_summary, metrics, _path_value(self.paths, "foundation_checkpoint"), evaluation_scope="pilot")
        self._record_result(key, record)
        self.state["artifacts"]["S0_checkpoint"] = _safe_basename_ref(checkpoint)
        self._save_state()
        return record

    def _run_joint(self, views: Mapping[str, str]) -> Optional[dict]:
        key = "J3-diagnostic"
        if self.resume and self._result(key):
            return self._result(key)
        resolved = self._resolved("J3-diagnostic", "J3")
        resolved["protocol_hash"] = canonical_json_hash({"canonical": sha256_file(self.canonical), "eval": sha256_file(self.eval_slice), "seed": self.config["seed"], "role": "diagnostic_joint"})
        train_summary = self._train_long_stage(resolved, views["J3_current"], views["J3_dev"], "J3-diagnostic", _path_value(self.paths, "foundation_checkpoint"))
        eval_summary, metrics = self._evaluate(resolved, train_summary, views["J3_eval"], "J3-diagnostic")
        record = _result_record("J3", "J3-diagnostic", resolved, train_summary, eval_summary, metrics, _path_value(self.paths, "foundation_checkpoint"), evaluation_scope="pilot")
        record["diagnostic_only"] = True
        self._record_result(key, record)
        return record

    def _run_s1(self, views: Mapping[str, str], s0_result: Optional[Mapping[str, Any]]) -> dict:
        output = {}
        s0_result = s0_result or self._result("S0-v3") or self._historical_result("S0-v3")
        methods = ("R-QPLSEG-S1", "R-QPLSEG-KD-S1")
        if not s0_result or s0_result.get("execution_status") != "COMPLETE":
            for method in methods:
                resolved = self._resolved(method.rsplit("-S1", 1)[0], "S1")
                output[method] = self._record_result(method, _not_run_record("S1", method, resolved, "audited S0-v3 checkpoint is not complete", "NOT_RUN_S0_INVALID"))
            return output
        s0_checkpoint = self._checkpoint_for_result(s0_result)
        # Required before any S1 optimizer update: isolate the effect of
        # evaluating the S0 checkpoint with the previous and expanded
        # vocabularies. This is inference-only and never retrains S1.
        self._run_s1_zero_vocab(views, s0_checkpoint)
        _, _, replay_path = self._build_memory("S0", views["S0_current"], [206], "S0_memory.json", "S1_replay.json")
        pl_path, pl_audit = self._prepare_teacher_pl(
            "S1", s0_checkpoint, self._resolved("S0-v3", "S0"),
            views["S1_teacher_source"], views["S0_calibration"], views["S1_current"],
            [206], [792],
        )
        if not pl_path:
            for method in methods:
                resolved = self._resolved(method.rsplit("-S1", 1)[0], "S1")
                output[method] = self._record_result(method, _not_run_record("S1", method, resolved, "qpl_segment_v2 produced no admissible segments", "NOT_RUN_PL_EMPTY"))
            return output
        split = self._partition("S1")
        qpl_view, _ = self._make_view(
            "S1_qpl_segment_current_train.json", "S1", "train", "cil",
            split["train_video_ids"], [206, 792], [792], [206], "train", "train", pl_path, True,
        )
        memory_version = _read_json(replay_path).get("memory_version")
        for method, use_kd in (("R-QPLSEG-S1", False), ("R-QPLSEG-KD-S1", True)):
            base_method = method.rsplit("-S1", 1)[0]
            resolved = self._resolved(base_method, "S1")
            resolved["protocol_hash"] = canonical_json_hash({
                "canonical": sha256_file(self.canonical),
                "eval": sha256_file(self.eval_slice),
                "seed": self.config["seed"],
                "stage": "S1",
                "method_family": "QPLSEG",
            })
            resolved["replay_memory_version"] = memory_version
            if self.resume and self._result(method):
                output[method] = self._result(method)
                continue
            self._event("train_start", run=method, steps=int(dict(resolved.get("training", {})).get("incremental_steps", 600)))
            try:
                train_summary = self._call_train(
                    resolved, qpl_view, method, s0_checkpoint, "stage_transfer",
                    int(dict(resolved.get("training", {})).get("incremental_steps", 600)), replay_path,
                    teacher_checkpoint=s0_checkpoint if use_kd else None,
                    teacher_active=[206],
                    teacher_resolved=self._resolved("S0-v3", "S0") if use_kd else None,
                )
            except Exception as exc:
                from cmot.train import KDAlignmentBlocked
                if not isinstance(exc, KDAlignmentBlocked):
                    raise
                output[method] = self._record_result(method, _blocked_kd_record(
                    "S1", method, resolved, s0_checkpoint, s0_checkpoint,
                    qpl_view, replay_path, exc,
                ))
                continue
            eval_summary, metrics = self._evaluate(resolved, train_summary, views["S1_eval"], method)
            output[method] = self._record_result(
                method,
                _result_record(
                    "S1", method, resolved, train_summary, eval_summary, metrics,
                    s0_checkpoint, s0_checkpoint if use_kd else None,
                    pl_path, pl_audit, memory_version,
                ),
            )
        return output

    def _run_s2(self, views: Mapping[str, str], s1_results: Mapping[str, Mapping[str, Any]]) -> dict:
        output = {}
        qpl_parent = s1_results.get("R-QPLSEG-S1") or self._result("R-QPLSEG-S1")
        kd_parent = s1_results.get("R-QPLSEG-KD-S1") or self._result("R-QPLSEG-KD-S1")
        if not qpl_parent or qpl_parent.get("execution_status") != "COMPLETE":
            resolved = self._resolved("R-QPLSEG", "S2")
            output["R-QPLSEG-S2"] = self._record_result("R-QPLSEG-S2", _not_run_record("S2", "R-QPLSEG-S2", resolved, "R-QPLSEG-S1 did not complete", "NOT_RUN_PARENT_QPLSEG_INVALID"))
            kd_resolved = self._resolved("R-QPLSEG-KD", "S2")
            output["R-QPLSEG-KD-S2"] = self._record_result("R-QPLSEG-KD-S2", _not_run_record("S2", "R-QPLSEG-KD-S2", kd_resolved, "R-QPLSEG-S1 did not complete", "NOT_RUN_PARENT_QPLSEG_INVALID"))
            self._run_s2_zero_vocab(views, s1_results)
            return output

        qpl_s1_checkpoint = self._checkpoint_for_result(qpl_parent)
        self._run_s2_zero_vocab(views, s1_results)
        _, _, s2_replay = self._build_memory("S1", views["S1_current"], [206, 792], "S1_memory_common.json", "S2_replay.json")
        pl_path, pl_audit = self._prepare_teacher_pl(
            "S2", qpl_s1_checkpoint, self._resolved("R-QPLSEG", "S1"),
            views["S2_teacher_source"], views["S2_calibration"], views["S2_current"],
            [206, 792], [1122],
        )
        memory_version = _read_json(s2_replay).get("memory_version")
        if not pl_path:
            resolved = self._resolved("R-QPLSEG", "S2")
            output["R-QPLSEG-S2"] = self._record_result("R-QPLSEG-S2", _not_run_record("S2", "R-QPLSEG-S2", resolved, "S2 qpl_segment_v2 produced no admissible car/pedestrian segments", "NOT_RUN_PL_EMPTY"))
            return output
        split = self._partition("S2")
        qpl_view, _ = self._make_view(
            "S2_qpl_segment_current_train.json", "S2", "train", "cil",
            split["train_video_ids"], ALL_IDS, [1122], [206, 792], "train", "train", pl_path, True,
        )
        resolved = self._resolved("R-QPLSEG", "S2")
        resolved["protocol_hash"] = canonical_json_hash({
            "canonical": sha256_file(self.canonical),
            "eval": sha256_file(self.eval_slice),
            "seed": self.config["seed"],
            "stage": "S2",
            "method_family": "QPLSEG",
        })
        resolved["replay_memory_version"] = memory_version
        if self.resume and self._result("R-QPLSEG-S2"):
            output["R-QPLSEG-S2"] = self._result("R-QPLSEG-S2")
        else:
            self._event("train_start", run="R-QPLSEG-S2", steps=int(dict(resolved.get("training", {})).get("s2_steps", 600)))
            train_summary = self._call_train(
                resolved, qpl_view, "R-QPLSEG-S2", qpl_s1_checkpoint, "stage_transfer",
                int(dict(resolved.get("training", {})).get("s2_steps", 600)), s2_replay,
            )
            eval_summary, metrics = self._evaluate(resolved, train_summary, views["S2_eval"], "R-QPLSEG-S2")
            output["R-QPLSEG-S2"] = self._record_result(
                "R-QPLSEG-S2",
                _result_record("S2", "R-QPLSEG-S2", resolved, train_summary, eval_summary, metrics, qpl_s1_checkpoint, None, pl_path, pl_audit, memory_version),
            )

        kd_resolved = self._resolved("R-QPLSEG-KD", "S2")
        kd_resolved["protocol_hash"] = resolved["protocol_hash"]
        kd_resolved["replay_memory_version"] = memory_version
        kd_valid = float(kd_parent.get("valid_kd_objects", 0.0) or 0.0) if kd_parent else 0.0
        if not kd_parent or kd_parent.get("execution_status") != "COMPLETE" or kd_valid <= 0.0:
            output["R-QPLSEG-KD-S2"] = self._record_result(
                "R-QPLSEG-KD-S2",
                _not_run_record(
                    "S2", "R-QPLSEG-KD-S2", kd_resolved,
                    "R-QPLSEG-KD-S1 did not complete with valid KD objects",
                    "NOT_RUN_PARENT_KD_INVALID",
                ),
            )
        elif self.resume and self._result("R-QPLSEG-KD-S2"):
            output["R-QPLSEG-KD-S2"] = self._result("R-QPLSEG-KD-S2")
        else:
            kd_s1_checkpoint = self._checkpoint_for_result(kd_parent)
            self._event("train_start", run="R-QPLSEG-KD-S2", steps=int(dict(kd_resolved.get("training", {})).get("s2_steps", 600)))
            try:
                train_summary = self._call_train(
                    kd_resolved, qpl_view, "R-QPLSEG-KD-S2", kd_s1_checkpoint, "stage_transfer",
                    int(dict(kd_resolved.get("training", {})).get("s2_steps", 600)), s2_replay,
                    teacher_checkpoint=kd_s1_checkpoint, teacher_active=[206, 792],
                    teacher_resolved=self._resolved("R-QPLSEG-KD", "S1"),
                )
            except Exception as exc:
                from cmot.train import KDAlignmentBlocked
                if not isinstance(exc, KDAlignmentBlocked):
                    raise
                output["R-QPLSEG-KD-S2"] = self._record_result(
                    "R-QPLSEG-KD-S2",
                    _blocked_kd_record("S2", "R-QPLSEG-KD-S2", kd_resolved, kd_s1_checkpoint, kd_s1_checkpoint, qpl_view, s2_replay, exc),
                )
            else:
                eval_summary, metrics = self._evaluate(kd_resolved, train_summary, views["S2_eval"], "R-QPLSEG-KD-S2")
                output["R-QPLSEG-KD-S2"] = self._record_result(
                    "R-QPLSEG-KD-S2",
                    _result_record("S2", "R-QPLSEG-KD-S2", kd_resolved, train_summary, eval_summary, metrics, kd_s1_checkpoint, kd_s1_checkpoint, pl_path, pl_audit, memory_version),
                )
        return output

    def _run_p3(self, views: Mapping[str, str], s2_results: Mapping[str, Mapping[str, Any]]) -> dict:
        """Run the single authorized plasticity-first S2 experiment, if gated."""
        method = "R-QPLSEG-PF-S2"
        resolved = self._resolved("R-QPLSEG-PF", "S2")
        existing = self._result(method)
        if self.resume and existing:
            return {method: existing}

        qpl_s2 = s2_results.get("R-QPLSEG-S2") or self._result("R-QPLSEG-S2")
        gate = {
            "retention_threshold": 0.95,
            "truck_gap_threshold": 0.02,
            "zero_step_old_HOTA_mean": None,
            "qpl_s2_old_HOTA_mean": None,
            "retention_ratio": None,
            "j3_truck_HOTA_mean": None,
            "qpl_s2_truck_HOTA_mean": None,
            "j3_minus_qpl_s2_truck_HOTA_mean": None,
            "retention_pass": False,
            "truck_gap_pass": False,
            "authorized": False,
        }
        if not qpl_s2 or qpl_s2.get("execution_status") != "COMPLETE":
            record = _not_run_record(
                "S2", method, resolved,
                "R-QPLSEG-S2 was not complete; plasticity gate was not evaluated",
                "NOT_RUN_PARENT_QPLSEG_INVALID",
            )
            record["plasticity_gate"] = gate
            return {method: self._record_result(method, record)}

        def _number(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if value == value else None

        zero = self._result("zero_S2_R-QPLSEG-S1_expanded_vocabulary")
        zero_metrics = zero.get("metrics", {}) if isinstance(zero, Mapping) else {}
        zero_old = zero_metrics.get("old_macro", {}) if isinstance(zero_metrics, Mapping) else {}
        qpl_metrics = qpl_s2.get("metrics", {})
        qpl_old = qpl_metrics.get("old_macro", {}) if isinstance(qpl_metrics, Mapping) else {}
        qpl_per_class = qpl_metrics.get("per_class", {}) if isinstance(qpl_metrics, Mapping) else {}
        gate["zero_step_old_HOTA_mean"] = _number(zero_old.get("HOTA_mean"))
        gate["qpl_s2_old_HOTA_mean"] = _number(qpl_old.get("HOTA_mean"))
        if gate["zero_step_old_HOTA_mean"] not in (None, 0.0) and gate["qpl_s2_old_HOTA_mean"] is not None:
            gate["retention_ratio"] = gate["qpl_s2_old_HOTA_mean"] / gate["zero_step_old_HOTA_mean"]
        qpl_truck = qpl_per_class.get("1122", {}) if isinstance(qpl_per_class, Mapping) else {}
        gate["qpl_s2_truck_HOTA_mean"] = _number(qpl_truck.get("HOTA_mean"))
        j3 = self._result("J3-diagnostic") or self._historical_result("J3-diagnostic")
        j3_metrics = j3.get("metrics", {}) if isinstance(j3, Mapping) else {}
        j3_per_class = j3_metrics.get("per_class", {}) if isinstance(j3_metrics, Mapping) else {}
        j3_truck = j3_per_class.get("1122", {}) if isinstance(j3_per_class, Mapping) else {}
        gate["j3_truck_HOTA_mean"] = _number(j3_truck.get("HOTA_mean"))
        if gate["j3_truck_HOTA_mean"] is not None and gate["qpl_s2_truck_HOTA_mean"] is not None:
            gate["j3_minus_qpl_s2_truck_HOTA_mean"] = gate["j3_truck_HOTA_mean"] - gate["qpl_s2_truck_HOTA_mean"]
        gate["retention_pass"] = bool(
            gate["retention_ratio"] is not None
            and gate["retention_ratio"] >= gate["retention_threshold"]
        )
        gate["truck_gap_pass"] = bool(
            gate["j3_minus_qpl_s2_truck_HOTA_mean"] is not None
            and gate["j3_minus_qpl_s2_truck_HOTA_mean"] >= gate["truck_gap_threshold"]
        )
        gate["authorized"] = bool(gate["retention_pass"] and gate["truck_gap_pass"])
        if not gate["authorized"]:
            record = _not_run_record(
                "S2", method, resolved,
                "stability/plasticity conditions were not both met",
                "NOT_RUN_CONDITION_NOT_MET",
            )
            record["plasticity_gate"] = gate
            record["zero_step_old_HOTA_mean"] = gate["zero_step_old_HOTA_mean"]
            record["retention_ratio"] = gate["retention_ratio"]
            return {method: self._record_result(method, record)}

        qpl_parent = self._result("R-QPLSEG-S1")
        qpl_view = self.manifest_root / "S2_qpl_segment_current_train.json"
        replay_view = self.memory_root / "S2_replay.json"
        pl_path = self.teacher_root / "S2_qpl_segment_v2.jsonl"
        admission_path = self.teacher_root / "S2_admission.json"
        if not qpl_parent or qpl_parent.get("execution_status") != "COMPLETE" or not qpl_view.is_file() or not replay_view.is_file() or not pl_path.is_file():
            record = _not_run_record(
                "S2", method, resolved,
                "shared QPLSEG-S2 current/replay artifacts were not available",
                "NOT_RUN_ARTIFACT_BINDING",
            )
            record["plasticity_gate"] = gate
            return {method: self._record_result(method, record)}
        parent_checkpoint = self._checkpoint_for_result(qpl_parent)
        memory_version = _read_json(str(replay_view)).get("memory_version")
        resolved["protocol_hash"] = canonical_json_hash({
            "canonical": sha256_file(self.canonical),
            "eval": sha256_file(self.eval_slice),
            "seed": self.config["seed"],
            "stage": "S2",
            "method_family": "QPLSEG",
            "plasticity_schedule": list(resolved.get("replay", {}).get("ratio_schedule", [])),
        })
        resolved["replay_memory_version"] = memory_version
        steps = int(dict(resolved.get("training", {})).get("s2_steps", 600))
        self._event("train_start", run=method, steps=steps)
        train_summary = self._call_train(
            resolved, str(qpl_view), method, parent_checkpoint, "stage_transfer", steps, str(replay_view)
        )
        eval_summary, metrics = self._evaluate(resolved, train_summary, views["S2_eval"], method)
        pl_audit = _read_json(str(admission_path)) if admission_path.is_file() else None
        record = _result_record(
            "S2", method, resolved, train_summary, eval_summary, metrics,
            parent_checkpoint, None, str(pl_path), pl_audit, memory_version,
        )
        record["plasticity_gate"] = gate
        record["zero_step_old_HOTA_mean"] = gate["zero_step_old_HOTA_mean"]
        record["retention_ratio"] = gate["retention_ratio"]
        return {method: self._record_result(method, record)}

    def _run_motion(self, views: Mapping[str, str], s1_results: Mapping[str, Mapping[str, Any]]) -> dict:
        candidates = [s1_results.get("R-QPL-KD-S1"), s1_results.get("R-ER-S1")]
        baseline = next((value for value in candidates if value and value.get("execution_status") == "COMPLETE" and (value.get("new_class_recall_at_hota_005") or 0.0) > 0.0), None)
        if baseline is None:
            result = {}
            for method in ("O1-residual-v2", "O2-residual-v2"):
                result[method] = self._record_result(method, {"method": method, "stage": "S1", "execution_status": "NOT_RUN", "evaluation_scope": "pilot", "reason": "baseline new-class recall was zero; motion branch not authorized", "optimizer_steps": 0})
            return result
        baseline_method = baseline["method"]
        baseline_checkpoint = self._checkpoint_path(baseline_method, int(baseline["optimizer_steps"]))
        # Motion branches use the same current/replay views as their selected
        # baseline; no extra data or pretraining is introduced.
        current = views["S1_current"]
        replay = str(self.memory_root / "S1_replay.json")
        output = {}
        for method in ("O1-residual-v2", "O2-residual-v2"):
            resolved = self._resolved(method, "S1")
            resolved["protocol_hash"] = canonical_json_hash({"canonical": sha256_file(self.canonical), "eval": sha256_file(self.eval_slice), "seed": self.config["seed"], "stage": "S1", "motion": method})
            resolved["replay_memory_version"] = _read_json(replay).get("memory_version")
            if self.resume and self._result(method):
                output[method] = self._result(method)
                continue
            train_summary = self._train_stage(resolved, current, method, baseline_checkpoint, replay)
            eval_summary, metrics = self._evaluate(resolved, train_summary, views["S1_eval"], method)
            output[method] = self._record_result("S1", method, resolved, train_summary, eval_summary, metrics, baseline_checkpoint, None, None, None, _read_json(replay).get("memory_version"))
        return output

    def _refresh_memory_bindings(self) -> None:
        """Repair/audit replay-version fields for already completed runs.

        Replay manifests store the immutable memory identity under
        ``memory_version``.  Keep this audit limited to basenames and hashes
        so it can safely flow into the public report without exposing the
        private run root.
        """
        replay_views = {
            "R-QPLSEG-S1": "S1_replay.json",
            "R-QPLSEG-KD-S1": "S1_replay.json",
            "R-QPLSEG-S2": "S2_replay.json",
            "R-QPLSEG-KD-S2": "S2_replay.json",
            "R-QPLSEG-PF-S2": "S2_replay.json",
        }
        changed = False
        for key, basename in replay_views.items():
            record = self.state.get("results", {}).get(key)
            if not isinstance(record, dict) or record.get("execution_status") != "COMPLETE":
                continue
            path = self.memory_root / basename
            if not path.is_file():
                continue
            payload = _read_json(str(path))
            memory_version = payload.get("memory_version")
            if not memory_version:
                continue
            updated = dict(record)
            updated["memory_version"] = memory_version
            updated["replay_view_sha256"] = sha256_file(str(path))
            updated["memory_binding_audit"] = {
                "manifest_basename": basename,
                "manifest_sha256": sha256_file(str(path)),
                "memory_version": memory_version,
            }
            if updated != record:
                self.state["results"][key] = updated
                changed = True
        if changed:
            self._save_state()

    def _write_public_reports(self) -> None:
        report_root = Path(__file__).resolve().parents[2] / "reports" / "v3_1"
        report_root.mkdir(parents=True, exist_ok=True)
        public_results = []
        for key, value in self.state.get("results", {}).items():
            item = dict(value)
            observed_steps = int(item.get("observed_optimizer_steps", 0) or item.get("optimizer_steps", 0) or 0)
            if observed_steps > 0 and not item.get("checkpoint"):
                candidate = self.run_root / str(item.get("method", key)) / ("checkpoint_%03d.pt" % observed_steps)
                if candidate.is_file():
                    item["checkpoint"] = {"basename": candidate.name, "sha256": sha256_file(str(candidate))}
                    if isinstance(item.get("observed_result"), dict):
                        item["observed_result"] = dict(item["observed_result"])
                        item["observed_result"]["checkpoint"] = item["checkpoint"]
                    self.state["results"][key]["checkpoint"] = item["checkpoint"]
                    if isinstance(self.state["results"][key].get("observed_result"), dict):
                        self.state["results"][key]["observed_result"]["checkpoint"] = item["checkpoint"]
            if int(item.get("optimizer_steps", 0) or 0) > 0 and not item.get("actual_loss_weights"):
                method = str(item.get("method", key))
                stage = str(item.get("stage", ""))
                resolve_method = method
                if method.endswith("-S1") or method.endswith("-S2"):
                    resolve_method = method.rsplit("-", 1)[0]
                try:
                    contract = self._actual_runtime_contract(self._resolved(resolve_method, stage))
                    item["actual_modules"] = dict(contract.get("modules", {}))
                    item["actual_loss_weights"] = dict(contract.get("loss_weights", {}))
                    self.state["results"][key].update({
                        "actual_modules": item["actual_modules"],
                        "actual_loss_weights": item["actual_loss_weights"],
                    })
                except Exception as exc:
                    item["actual_modules"] = item.get("modules", {})
                    item["actual_loss_weights"] = "NOT_RUN_runtime_contract_audit:%s" % type(exc).__name__
            # Private artifact paths never cross the report boundary.
            for key in ("pl_path", "diagnosis_file"):
                item.pop(key, None)
            public_results.append(item)
        historical_methods = (
            "S0-v3", "J3-diagnostic", "R-ER-S1", "R-QPL-S1", "R-ER-S2",
        )
        historical_references = {}
        for method in historical_methods:
            value = self._historical_result(method)
            if value is None:
                continue
            historical_references[method] = {
                "method": value.get("method"),
                "stage": value.get("stage"),
                "execution_status": value.get("execution_status"),
                "evaluation_scope": value.get("evaluation_scope"),
                "optimizer_steps": value.get("optimizer_steps", 0),
                "checkpoint": value.get("checkpoint"),
                "metrics": value.get("metrics"),
                "valid_kd_objects": value.get("valid_kd_objects", "NOT_RUN"),
            }
        historical_zero = []
        for method, value in sorted(self.historical_results.items()):
            if not str(method).startswith("zero_"):
                continue
            historical_zero.append({
                "method": value.get("method"),
                "stage": value.get("stage"),
                "execution_status": value.get("execution_status"),
                "optimizer_steps": value.get("optimizer_steps", 0),
                "vocabulary_role": value.get("vocabulary_role"),
                "metrics": value.get("metrics"),
                "checkpoint": value.get("checkpoint"),
            })
        self._save_state()
        write_json(str(report_root / "results.json"), {
            "schema_version": "cmot.continual_v3_1.public_results.v1",
            "config": {"basename": Path(self.config_path).name, "sha256": sha256_file(self.config_path)},
            "canonical_manifest_sha256": sha256_file(self.canonical),
            "immutable_eval_manifest_sha256": sha256_file(self.eval_slice),
            "source_audit": dict(self.runtime.get("source_audit", {})),
            "download_audit": {
                "status": "NOT_RUN_no_new_data_or_dependency_download",
                "proxy_route_check": "NOT_RUN_no_download_requested",
            },
            "historical_references": {
                "methods": historical_references,
                "zero_step": historical_zero,
            },
            "results": public_results,
        })
        rows = []
        for value in public_results:
            metrics = value.get("metrics", {}) if isinstance(value.get("metrics"), dict) else {}
            all_seen = metrics.get("all_seen_macro", {}) if isinstance(metrics.get("all_seen_macro"), dict) else {}
            old = metrics.get("old_macro", {}) if isinstance(metrics.get("old_macro"), dict) else {}
            new = metrics.get("new_macro", {}) if isinstance(metrics.get("new_macro"), dict) else {}
            rows.append({
                "stage": value.get("stage"), "method": value.get("method"), "protocol_role": value.get("protocol_role"),
                "execution_status": value.get("execution_status"), "evaluation_scope": value.get("evaluation_scope"),
                "optimizer_steps": value.get("optimizer_steps", 0), "observed_optimizer_steps": value.get("observed_optimizer_steps"),
                "zero_step": value.get("zero_step", False), "vocabulary_role": value.get("vocabulary_role", "NOT_RUN"),
                "new_class_recall_at_hota_005": value.get("new_class_recall_at_hota_005"),
                "old_HOTA_mean": old.get("HOTA_mean", "NOT_RUN"), "old_IDF1": old.get("IDF1", "NOT_RUN"), "old_MOTA": old.get("MOTA", "NOT_RUN"),
                "new_HOTA_mean": new.get("HOTA_mean", "NOT_RUN"), "new_IDF1": new.get("IDF1", "NOT_RUN"), "new_MOTA": new.get("MOTA", "NOT_RUN"),
                "all_seen_HOTA_mean": all_seen.get("HOTA_mean", "NOT_RUN"), "all_seen_IDF1": all_seen.get("IDF1", "NOT_RUN"), "all_seen_MOTA": all_seen.get("MOTA", "NOT_RUN"),
                "valid_kd_objects": value.get("valid_kd_objects", "NOT_RUN"),
                "kd_replay_batches_seen": value.get("kd_replay_batches_seen", "NOT_RUN"),
                "kd_replay_gt_objects": (value.get("kd_diagnostics", {}) or {}).get("replay_gt_objects", "NOT_RUN") if isinstance(value.get("kd_diagnostics"), dict) else "NOT_RUN",
                "kd_student_iou_pass": (value.get("kd_diagnostics", {}) or {}).get("student_iou_pass", "NOT_RUN") if isinstance(value.get("kd_diagnostics"), dict) else "NOT_RUN",
                "kd_teacher_iou_pass": (value.get("kd_diagnostics", {}) or {}).get("teacher_iou_pass", "NOT_RUN") if isinstance(value.get("kd_diagnostics"), dict) else "NOT_RUN",
                "kd_joint_alignment_pass": (value.get("kd_diagnostics", {}) or {}).get("joint_alignment_pass", "NOT_RUN") if isinstance(value.get("kd_diagnostics"), dict) else "NOT_RUN",
                "kd_shared_old_class_objects": (value.get("kd_diagnostics", {}) or {}).get("shared_old_class_objects", "NOT_RUN") if isinstance(value.get("kd_diagnostics"), dict) else "NOT_RUN",
                "kd_teacher_score_pass": (value.get("kd_diagnostics", {}) or {}).get("teacher_score_pass", "NOT_RUN") if isinstance(value.get("kd_diagnostics"), dict) else "NOT_RUN",
                "kd_cells": (value.get("kd_diagnostics", {}) or {}).get("kd_cells", "NOT_RUN") if isinstance(value.get("kd_diagnostics"), dict) else "NOT_RUN",
                "checkpoint_sha256": (value.get("checkpoint") or {}).get("sha256") if isinstance(value.get("checkpoint"), dict) else value.get("model_sha256"),
                "parent_checkpoint_sha256": value.get("parent_checkpoint_sha256"),
                "teacher_checkpoint_sha256": value.get("teacher_checkpoint_sha256"),
                "current_view_sha256": value.get("current_view_sha256"),
                "replay_view_sha256": value.get("replay_view_sha256"),
                "pl_manifest_sha256": value.get("pl_manifest_sha256"),
                "raw_prediction_sha256": (value.get("raw_prediction") or {}).get("sha256") if isinstance(value.get("raw_prediction"), dict) else value.get("raw_prediction"),
                "raw_metrics_sha256": (value.get("raw_metrics") or {}).get("sha256") if isinstance(value.get("raw_metrics"), dict) else value.get("raw_metrics"),
                "reason": value.get("reason", ""),
            })
        fields = list(rows[0].keys()) if rows else ["stage", "method", "execution_status"]
        with (report_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        resolved_public = {
            "config": {"basename": Path(self.config_path).name, "sha256": sha256_file(self.config_path)},
            "source_audit": dict(self.runtime.get("source_audit", {})),
            "download_audit": {
                "status": "NOT_RUN_no_new_data_or_dependency_download",
                "proxy_route_check": "NOT_RUN_no_download_requested",
            },
            "experiments": {},
        }
        for key, value in self.state.get("results", {}).items():
            resolved_public["experiments"][key] = {
                "method": value.get("method"), "stage": value.get("stage"), "protocol_role": value.get("protocol_role"),
                "execution_status": value.get("execution_status"), "optimizer_steps": value.get("optimizer_steps", 0),
                "modules": value.get("modules", {}), "inference_thresholds": value.get("inference_thresholds", {}),
                "actual_modules": value.get("actual_modules", value.get("modules", {})),
                "actual_loss_weights": value.get("actual_loss_weights", "NOT_RUN"),
                "checkpoint": value.get("checkpoint", {"basename": None, "sha256": value.get("model_sha256")}),
                "parent_checkpoint_sha256": value.get("parent_checkpoint_sha256"),
                "teacher_checkpoint_sha256": value.get("teacher_checkpoint_sha256"),
                "current_view_sha256": value.get("current_view_sha256"),
                "replay_view_sha256": value.get("replay_view_sha256"),
                "pl_manifest_sha256": value.get("pl_manifest_sha256"),
                "memory_version": value.get("memory_version"),
                "sampler_plan_hash": value.get("sampler_plan_hash"),
                "exposure": value.get("exposure", {}),
                "teacher_admission": value.get("teacher_admission", "NOT_RUN"),
                "kd_replay_batches_seen": value.get("kd_replay_batches_seen", "NOT_RUN"),
                "kd_diagnostics": value.get("kd_diagnostics", "NOT_RUN"),
                "valid_kd_objects": value.get("valid_kd_objects", "NOT_RUN"),
                "plasticity_gate": value.get("plasticity_gate", "NOT_RUN"),
                "zero_step_old_HOTA_mean": value.get("zero_step_old_HOTA_mean"),
                "retention_ratio": value.get("retention_ratio"),
                "zero_step": value.get("zero_step", False),
                "vocabulary_role": value.get("vocabulary_role"),
                "evaluation_vocabulary_override": value.get("evaluation_vocabulary_override"),
                "resolved_config_sha256": value.get("resolved_config_sha256"),
            }
        write_json(str(report_root / "resolved_configs_sanitized.json"), resolved_public)
        lines = [
            "# C-MOT V3.1 修复与实验摘要",
            "",
            "本报告只引用脱敏哈希、basename和真实执行状态；未运行项保留为 `NOT_RUN`。",
            "",
            "- 公共执行路径接入了 V3.1 配置解析、stable video/frame/track UID、分来源 GT/PL 监督、质量门控 PL、一对一 PL 匹配、合法 GT 回放、跨来源分层采样、推理阈值分离、重复查询抑制和 checkpoint→prediction→TrackEval 绑定。",
            "- 本轮 motion 保持关闭；未声明 GRU、MoE、LoRA、Prompt或新检测器已运行。",
            "- J3 仅为 `diagnostic_joint`，不向 CIL 提供 checkpoint、PL、memory或校准参数。",
            "- 原始数据、标注、权重、完整预测和服务器私有路径均未写入仓库。",
            "- 下载审计：`%s`；代理路径检查：`%s`。本轮没有新数据或大依赖下载。" % (
                self.runtime.get("source_audit", {}).get("downloads", "NOT_RUN"),
                self.runtime.get("source_audit", {}).get("route_audit", "NOT_RUN"),
            ),
            "- canonical/eval manifest SHA256：`%s` / `%s`。" % (sha256_file(self.canonical), sha256_file(self.eval_slice)),
            "",
            "## 真实状态",
            "",
        ]
        for value in public_results:
            lines.append("- `%s/%s`: `%s`, %s steps, scope `%s`." % (value.get("stage"), value.get("method"), value.get("execution_status"), value.get("optimizer_steps", 0), value.get("evaluation_scope", "NOT_RUN")))
            if value.get("zero_step"):
                lines.append("  - zero-step vocabulary=`%s`, override IDs=%s." % (value.get("vocabulary_role"), value.get("evaluation_vocabulary_override", {}).get("evaluation_active_global_ids", "NOT_RUN")))
            admission = value.get("teacher_admission")
            if isinstance(admission, dict):
                selected_by_class = admission.get(
                    "selected_segments_by_class",
                    admission.get("accepted_by_class", {}),
                )
                lines.append(
                    "  - PL admission=%s, selected_segments_by_class=%s, "
                    "selected_frame_predictions=%s, disabled=%s."
                    % (
                        admission.get("status"),
                        selected_by_class,
                        admission.get("selected_frame_predictions", "NOT_RUN"),
                        admission.get("disabled_global_ids", []),
                    )
                )
            if value.get("observed_optimizer_steps") is not None:
                lines.append("  - observed redundant run steps=%s; it is not counted as an independent result." % value.get("observed_optimizer_steps"))
        (report_root / "changes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        failures = [value for value in public_results if value.get("execution_status") not in ("COMPLETE",)]
        failure_lines = ["# C-MOT V3.1 失败与限制记录", "", "只保留实际状态，不用估计值补齐。", ""]
        if not failures:
            failure_lines.append("本轮控制器未记录失败或等价跳过状态；这不等同于所有指标为正。")
        else:
            for value in failures:
                failure_lines.append("- `%s/%s`: `%s`；原因：%s" % (value.get("stage"), value.get("method"), value.get("execution_status"), value.get("reason", "NOT_RUN")))
                if value.get("observed_optimizer_steps") is not None:
                    failure_lines.append("  - observed_optimizer_steps=%s，原始产物仅作为等价审计证据。" % value.get("observed_optimizer_steps"))
        failure_lines.extend([
            "",
            "KD 是否有效以每个新命名 QPLSEG run 的实际 gate 统计为准；若被 fail-fast 阻断，保留具体 gate 状态，不改写成等价结果。",
            "新类与全部已见类指标均按 TrackEval 原始输出记录；负 MOTA 或无提升不作阈值修饰。",
        ])
        (report_root / "failure_analysis_zh.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    def run(self) -> dict:
        self._event("start", targets=list(self.targets), resume=self.resume)
        views = self._prepare_views()
        if "diagnose" in self.targets:
            self._diagnose()
        # V3.1 consumes the immutable, SHA-verified historical S0/J3
        # artifacts.  They are never retrained by the default target set.
        s0_result = self._result("S0-v3") or self._historical_result("S0-v3")
        joint_result = self._result("J3-diagnostic") or self._historical_result("J3-diagnostic")
        if "s0" in self.targets:
            try:
                s0_result = self._run_s0(views)
            except Exception as exc:
                self._event("failure", target="s0", type=type(exc).__name__, message=str(exc))
                s0_result = None
                self._record_result("S0-v3", {"method": "S0-v3", "stage": "S0", "execution_status": "FAILED", "evaluation_scope": "pilot", "reason": str(exc), "optimizer_steps": 0})
        if "joint" in self.targets:
            try:
                joint_result = self._run_joint(views)
            except Exception as exc:
                self._event("failure", target="joint", type=type(exc).__name__, message=str(exc))
                self._record_result("J3-diagnostic", {"method": "J3-diagnostic", "stage": "J3", "execution_status": "FAILED", "evaluation_scope": "pilot", "reason": str(exc), "optimizer_steps": 0})
        s1_results = {}
        if "s1" in self.targets:
            try:
                s1_results = self._run_s1(views, s0_result)
            except Exception as exc:
                self._event("failure", target="s1", type=type(exc).__name__, message=str(exc))
                for method in ("R-QPLSEG-S1", "R-QPLSEG-KD-S1"):
                    if not self._result(method):
                        s1_results[method] = self._record_result(method, {"method": method, "stage": "S1", "execution_status": "FAILED", "evaluation_scope": "pilot", "reason": str(exc), "optimizer_steps": 0})
        s2_results = {}
        if "s2" in self.targets:
            try:
                s2_results = self._run_s2(views, s1_results or {key: self._result(key) for key in ("R-QPLSEG-S1", "R-QPLSEG-KD-S1") if self._result(key)})
            except Exception as exc:
                self._event("failure", target="s2", type=type(exc).__name__, message=str(exc))
                for method in ("R-QPLSEG-S2", "R-QPLSEG-KD-S2"):
                    if not self._result(method):
                        self._record_result(method, {"method": method, "stage": "S2", "execution_status": "FAILED", "evaluation_scope": "pilot", "reason": str(exc), "optimizer_steps": 0})
        if "p3" in self.targets:
            try:
                self._run_p3(views, s2_results or {key: self._result(key) for key in ("R-QPLSEG-S2",) if self._result(key)})
            except Exception as exc:
                self._event("failure", target="p3", type=type(exc).__name__, message=str(exc))
                if not self._result("R-QPLSEG-PF-S2"):
                    resolved = self._resolved("R-QPLSEG-PF", "S2")
                    self._record_result("R-QPLSEG-PF-S2", {"method": "R-QPLSEG-PF-S2", "stage": "S2", "execution_status": "FAILED", "evaluation_scope": "pilot", "reason": str(exc), "optimizer_steps": 0, "protocol_role": resolved.get("protocol_role")})
        if "motion" in self.targets:
            try:
                self._run_motion(views, s1_results or {key: self._result(key) for key in ("R-ER-S1", "R-QPL-KD-S1") if self._result(key)})
            except Exception as exc:
                self._event("failure", target="motion", type=type(exc).__name__, message=str(exc))
        self._refresh_memory_bindings()
        self._write_public_reports()
        self._event("complete", result_count=len(self.state.get("results", {})))
        self._write_public_reports()
        return {"status": "COMPLETE", "targets": list(self.targets), "results": self.state.get("results", {})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runner = V3Runner(args.config, args.runtime, args.targets.split(","), resume=args.resume)
    if not args.execute:
        print(json.dumps({"status": "VALIDATED_NOT_EXECUTED", "targets": list(runner.targets)}, sort_keys=True))
        return
    print(json.dumps(runner.run(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
