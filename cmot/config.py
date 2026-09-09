"""Strict loaders and runtime resolution for C-MOT curricula."""

from pathlib import Path
from typing import Any, Mapping

import yaml


ALLOWED_KEYS = {
    "project", "schema_version", "seed", "network", "resources", "protocol",
    "training", "motion", "replay", "report", "supervision", "pseudo",
    "calibration", "distillation", "inference", "evaluation", "diagnosis",
}

V2_NESTED_KEYS = {
    "network": {"bulk_policy", "require_route_audit", "fallback_to_proxy", "initial_download_budget_gib"},
    "resources": {"protected_gpu_ids", "max_gpus", "max_concurrent_trainers", "dataloader_workers_per_trainer", "min_free_ram_fraction"},
    "protocol": {"name", "class_order", "train_source_cap", "pilot_train_source_cap_per_stage", "eval_video_cap", "eval_manifest_is_immutable", "train_split", "eval_split"},
    "training": {"batch_size_per_gpu", "clip_frames", "workers", "s0_steps", "incremental_steps", "lr_backbone", "lr_heads", "lr_motion", "weight_decay", "max_grad_norm", "input_size", "early_eval_optimizer_steps", "max_optimizer_steps_per_stage", "smoke_optimizer_steps", "custom_activation_checkpoint", "implicit_downloads"},
    "motion": {"enabled", "history_length", "hidden_dim", "num_modes", "mode", "implementation", "velocity_limit", "detach_features", "detach_reference", "detach_motion_reference", "warmup_steps", "lambda_motion", "max_dt"},
    "replay": {"budget_mib", "source", "ratio_schedule", "clip_len"},
    "report": {"require_raw_metrics", "allow_estimated_metrics", "redact_private_paths"},
    "supervision": {"pl_score_threshold", "conflict_iou", "pl_duplicate_iou"},
    "inference": {"score_threshold", "filter_threshold", "miss_tolerance", "duplicate_iou", "inference_dedup_enabled", "maximum_quantity"},
    "evaluation": {"manifest", "classes"},
}

V3_NESTED_KEYS = {
    "network": {"bulk_policy", "require_route_audit", "fallback_to_proxy"},
    "resources": {"protected_gpu_ids", "max_gpus", "max_concurrent_trainers", "dataloader_workers_per_trainer"},
    "protocol": {"name", "class_order", "train_source_cap", "eval_video_cap", "eval_manifest_is_immutable", "train_split", "eval_split", "dev_source_cap", "calibration_source_cap"},
    "training": {"batch_size_per_gpu", "clip_frames", "workers", "s0_steps", "joint_steps", "incremental_steps", "s2_steps", "early_eval_optimizer_steps", "extension_steps", "max_optimizer_steps_per_stage", "lr_backbone", "lr_heads", "lr_motion", "weight_decay", "max_grad_norm", "input_size", "gradient_audit_every"},
    "motion": {"enabled", "implementation", "mode", "velocity_limit", "detach_features", "detach_reference", "warmup_steps", "lambda_motion", "max_dt"},
    "replay": {"budget_mib", "training_budget_mib", "calibration_budget_mib", "ratio_schedule", "clip_len", "negative_fraction"},
    "supervision": {"lambda_gt", "lambda_pl", "pl_warmup_steps", "quality_iou_min", "conflict_iou", "pl_duplicate_iou"},
    "pseudo": {
        "enabled", "policy_version", "per_class_threshold", "per_class_cap", "total_cap",
        "total_segment_cap", "per_class_segment_cap", "max_segment_frames", "total_frame_cap",
        "pl_clip_fraction", "min_segment_frames", "max_gap_s", "calibration_version",
    },
    "calibration": {"thresholds", "min_predictions", "min_videos", "wilson_lower_min", "iou_min", "source_split"},
    "distillation": {"enabled", "temperature", "lambda_kd", "warmup_steps", "iou_min", "score_min", "cache_enabled", "fail_fast_replay_batches"},
    "inference": {"birth_threshold", "keep_threshold", "export_threshold", "miss_tolerance", "duplicate_iou", "duplicate_feature_cos", "maximum_quantity", "dedup_new_new", "dedup_new_track", "merge_existing_ids", "inference_dedup_enabled"},
    "evaluation": {"manifest", "classes", "thresholds", "primary_metric"},
    "report": {"redact_private_paths", "allow_estimated_metrics", "require_raw_metrics"},
    "diagnosis": {"max_videos", "max_frames_per_video", "visualization_frames"},
}


def _validate_nested(value: dict, allowed: Mapping[str, set], label: str) -> None:
    for name, keys in allowed.items():
        section = value.get(name, {})
        if not isinstance(section, dict):
            raise ValueError("%s section %s must be a mapping" % (label, name))
        unknown = set(section) - keys
        if unknown:
            raise ValueError("unknown keys in %s %s: %s" % (label, name, sorted(unknown)))


def _validate_v2(value: dict) -> None:
    _validate_nested(value, V2_NESTED_KEYS, "repair_v2")
    motion = value.get("motion", {})
    if motion.get("implementation", "none") not in ("none", "one_step_residual_v2"):
        raise ValueError("repair_v2 motion implementation is not connected")
    if motion.get("mode", "none") not in ("none", "one_step_agnostic_v2", "one_step_conditioned_v2"):
        raise ValueError("repair_v2 motion mode is not connected")


def _validate_v3(value: dict) -> None:
    _validate_nested(value, V3_NESTED_KEYS, "continual_v3")
    motion = value.get("motion", {})
    if motion.get("implementation", "none") not in ("none", "one_step_residual_v2"):
        raise ValueError("continual_v3 motion implementation is not connected")
    if motion.get("mode", "none") not in ("none", "one_step_agnostic_v2", "one_step_conditioned_v2"):
        raise ValueError("continual_v3 motion mode is not connected")
    inference = value.get("inference", {})
    required = ("birth_threshold", "keep_threshold", "export_threshold", "duplicate_feature_cos")
    missing = [key for key in required if key not in inference]
    if missing:
        raise ValueError("continual_v3 inference contract missing: %s" % ",".join(missing))
    if not 0 <= float(inference["birth_threshold"]) <= 1 or not 0 <= float(inference["export_threshold"]) <= 1:
        raise ValueError("continual_v3 inference thresholds must be probabilities")


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("curriculum config must be a mapping")
    unknown = set(value) - ALLOWED_KEYS
    if unknown:
        raise ValueError("unknown top-level config keys: %s" % sorted(unknown))
    if value.get("project") != "C-MOT":
        raise ValueError("project must be C-MOT")
    schema = value.get("schema_version")
    if schema == "cmot.repair_v2":
        _validate_v2(value)
    elif schema == "cmot.continual_v3":
        _validate_v3(value)
    elif schema == "cmot.curriculum.v1":
        _validate_nested(value, V2_NESTED_KEYS, "curriculum")
    else:
        raise ValueError("unsupported curriculum schema_version")
    return value


def _copy_sections(curriculum: dict) -> dict:
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in curriculum.items()
    }


def _stage_ids(stage: str) -> dict:
    values = {
        "S0": {"active_global_ids": [206], "new_global_ids": [206], "old_global_ids": [], "label_mode": "complete", "protocol_role": "cil"},
        "J3": {"active_global_ids": [206, 792, 1122], "new_global_ids": [206, 792, 1122], "old_global_ids": [], "label_mode": "complete", "protocol_role": "diagnostic_joint"},
        "S1": {"active_global_ids": [206, 792], "new_global_ids": [792], "old_global_ids": [206], "label_mode": "partial", "protocol_role": "cil"},
        "S2": {"active_global_ids": [206, 792, 1122], "new_global_ids": [1122], "old_global_ids": [206, 792], "label_mode": "partial", "protocol_role": "cil"},
    }
    if stage not in values:
        raise ValueError("continual_v3 stage must be S0, J3, S1 or S2")
    return dict(values[stage])


def _method_flags(method: str, stage: str) -> dict:
    methods = {
        "S0-v3": (False, False, False),
        "J3-diagnostic": (False, False, False),
        "R-ER": (False, False, False),
        "R-QPL": (True, False, False),
        "R-QPL-KD": (True, True, False),
        "R-QPLSEG": (True, False, False),
        "R-QPLSEG-KD": (True, True, False),
        "R-QPLSEG-PF": (True, False, False),
        "R-QPLFRAME": (True, False, False),
        "R-QPLSEG-PARENT-ER": (False, False, False),
        "R-QPLSEG-CURRKD": (True, True, False),
        "R-QPLSEG-KDPARENT-NOKD": (True, False, False),
        "S2-R-ER": (False, False, False),
        "S2-R-QPL-KD": (True, True, False),
        "R-ER-S2": (False, False, False),
        "R-QPL-KD-S2": (True, True, False),
        "O1-residual-v2": (True, False, True),
        "O2-residual-v2": (True, True, True),
    }
    if method not in methods:
        raise ValueError("unknown continual_v3 method %s" % method)
    enable_pl, enable_kd, enable_motion = methods[method]
    if stage in ("S0", "J3"):
        enable_pl = enable_kd = enable_motion = False
    return {"enable_pl": enable_pl, "enable_kd": enable_kd, "enable_motion": enable_motion}


def resolve_runtime_config(curriculum: dict, runtime_paths: Mapping[str, Any], method: str, stage: str) -> dict:
    """Resolve all values consumed by the V3 dataset/model/trainer/evaluator."""
    schema = curriculum.get("schema_version")
    if schema == "cmot.repair_v2":
        _validate_v2(curriculum)
    elif schema == "cmot.continual_v3":
        _validate_v3(curriculum)
    else:
        raise ValueError("resolve_runtime_config requires a V2 or V3 curriculum")
    resolved = _copy_sections(curriculum)
    resolved["runtime_paths"] = dict(runtime_paths)
    resolved["method"] = str(method)
    resolved["stage"] = str(stage)
    resolved.update(_stage_ids(str(stage)))
    flags = _method_flags(str(method), str(stage))
    resolved.update(flags)
    motion = dict(resolved.get("motion", {}))
    if not flags["enable_motion"]:
        motion["implementation"] = "none"
        motion["mode"] = "none"
    elif str(method).startswith("O1"):
        motion["implementation"] = "one_step_residual_v2"
        motion["mode"] = "one_step_agnostic_v2"
    else:
        motion["implementation"] = "one_step_residual_v2"
        motion["mode"] = "one_step_conditioned_v2"
    resolved["motion"] = motion
    inference = dict(resolved.get("inference", {}))
    # V2 names remain readable, but V3 always materializes the new contract.
    inference.setdefault("birth_threshold", inference.get("score_threshold", 0.50))
    inference.setdefault("keep_threshold", inference.get("filter_threshold", 0.20))
    inference.setdefault("export_threshold", inference.get("score_threshold", 0.50))
    inference.setdefault("duplicate_feature_cos", 0.95)
    inference.setdefault("dedup_new_new", True)
    inference.setdefault("dedup_new_track", True)
    inference.setdefault("merge_existing_ids", False)
    inference.setdefault("inference_dedup_enabled", True)
    inference.setdefault("maximum_quantity", 160)
    inference.setdefault("miss_tolerance", 5)
    inference.setdefault("duplicate_iou", 0.85)
    resolved["inference"] = inference
    training = dict(resolved.get("training", {}))
    training.setdefault("joint_steps", training.get("s0_steps", 1200))
    training.setdefault("s2_steps", training.get("incremental_steps", 600))
    training.setdefault("early_eval_optimizer_steps", 600)
    training.setdefault("gradient_audit_every", 100)
    resolved["training"] = training
    supervision = dict(resolved.get("supervision", {}))
    supervision.setdefault("lambda_gt", 1.0)
    supervision.setdefault("lambda_pl", 0.25)
    supervision.setdefault("pl_warmup_steps", 100)
    supervision.setdefault("quality_iou_min", 0.5)
    supervision.setdefault("conflict_iou", 0.7)
    supervision.setdefault("pl_duplicate_iou", 0.7)
    # Method flags are part of the runtime contract.  R-ER must not receive
    # PL merely because the common YAML contains the QPL coefficient.
    supervision["configured_lambda_pl"] = float(supervision.get("lambda_pl", 0.25))
    supervision["lambda_pl"] = float(supervision.get("lambda_pl", 0.25)) if flags["enable_pl"] else 0.0
    resolved["supervision"] = supervision
    replay = dict(resolved.get("replay", {}))
    replay.setdefault("ratio_schedule", ["current", "replay"])
    replay.setdefault("clip_len", training.get("clip_frames", 4))
    if str(method) == "R-QPLSEG-PF":
        replay["ratio_schedule"] = ["current", "current", "current", "replay"]
    resolved["replay"] = replay
    pseudo = dict(resolved.get("pseudo", {}))
    pseudo.setdefault("policy_version", "qpl_segment_v2")
    pseudo.setdefault("total_segment_cap", 40)
    pseudo.setdefault("per_class_segment_cap", {})
    pseudo.setdefault("max_segment_frames", 8)
    pseudo.setdefault("total_frame_cap", 256)
    pseudo.setdefault("pl_clip_fraction", 0.25)
    pseudo.setdefault("min_segment_frames", 3)
    pseudo.setdefault("max_gap_s", 1.0)
    resolved["pseudo"] = pseudo
    calibration = dict(resolved.get("calibration", {}))
    calibration.setdefault("thresholds", [0.5, 0.6, 0.7, 0.8, 0.9])
    calibration.setdefault("min_predictions", 30)
    calibration.setdefault("min_videos", 3)
    calibration.setdefault("wilson_lower_min", 0.8)
    calibration.setdefault("iou_min", 0.5)
    resolved["calibration"] = calibration
    distillation = dict(resolved.get("distillation", {}))
    distillation.setdefault("temperature", 2.0)
    distillation.setdefault("lambda_kd", 0.25)
    distillation.setdefault("warmup_steps", 100)
    distillation.setdefault("iou_min", 0.5)
    distillation.setdefault("score_min", 0.5)
    distillation.setdefault("fail_fast_replay_batches", 8)
    distillation["enabled"] = bool(distillation.get("enabled", True)) and bool(flags["enable_kd"])
    resolved["distillation"] = distillation
    resolved["actual_consumers"] = {
        "protocol_role": resolved["protocol_role"],
        "dataset": {"clip_frames": training.get("clip_frames", 4), "input_size": training.get("input_size", [640, 360]), "workers": training.get("workers", 0)},
        "optimizer": {key: training.get(key) for key in ("lr_backbone", "lr_heads", "lr_motion", "weight_decay", "max_grad_norm")},
        "sampler": {"ratio_schedule": replay.get("ratio_schedule"), "negative_fraction": replay.get("negative_fraction", 0.2), "seed": resolved.get("seed")},
        "supervision": supervision,
        "pseudo": dict(resolved.get("pseudo", {})),
        "calibration": calibration,
        "distillation": distillation,
        "inference": inference,
        "evaluation": dict(resolved.get("evaluation", {})),
        "flags": flags,
    }
    return resolved
