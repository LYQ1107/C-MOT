"""Strict loader for the public C-MOT curriculum configuration."""

from pathlib import Path
from typing import Any, Mapping

import yaml


ALLOWED_KEYS = {
    "project", "schema_version", "seed", "network", "resources", "protocol",
    "training", "motion", "replay", "report", "supervision", "inference",
    "evaluation",
}
NESTED_KEYS = {
    "network": {"bulk_policy", "require_route_audit", "fallback_to_proxy", "initial_download_budget_gib"},
    "resources": {"protected_gpu_ids", "max_gpus", "max_concurrent_trainers", "dataloader_workers_per_trainer", "min_free_ram_fraction"},
    "protocol": {"name", "class_order", "pilot_train_source_cap_per_stage", "eval_manifest_is_immutable"},
    "training": {"batch_size_per_gpu", "clip_frames", "early_eval_optimizer_steps", "max_optimizer_steps_per_stage", "smoke_optimizer_steps", "custom_activation_checkpoint", "implicit_downloads"},
    "motion": {"enabled", "history_length", "hidden_dim", "num_modes", "mode", "detach_motion_reference", "warmup_steps", "lambda_motion"},
    "replay": {"budget_mib", "source", "ratio_schedule"},
    "report": {"require_raw_metrics", "allow_estimated_metrics", "redact_private_paths"},
}


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
    if value.get("schema_version") not in ("cmot.curriculum.v1", "cmot.repair_v2"):
        raise ValueError("unsupported curriculum schema_version")
    if value.get("schema_version") == "cmot.repair_v2":
        _validate_repair_v2(value)
        return value
    for name, allowed in NESTED_KEYS.items():
        section = value.get(name, {})
        if not isinstance(section, dict):
            raise ValueError("config section %s must be a mapping" % name)
        unknown_nested = set(section) - allowed
        if unknown_nested:
            raise ValueError("unknown keys in %s: %s" % (name, sorted(unknown_nested)))
    return value


REPAIR_V2_NESTED_KEYS = {
    "network": {"bulk_policy", "require_route_audit", "fallback_to_proxy"},
    "resources": {"protected_gpu_ids", "max_gpus", "max_concurrent_trainers", "dataloader_workers_per_trainer"},
    "protocol": {"name", "class_order", "train_source_cap", "eval_video_cap", "eval_manifest_is_immutable", "train_split", "eval_split"},
    "training": {"batch_size_per_gpu", "clip_frames", "workers", "s0_steps", "incremental_steps", "lr_backbone", "lr_heads", "lr_motion", "weight_decay", "max_grad_norm", "input_size"},
    "motion": {"enabled", "implementation", "mode", "velocity_limit", "detach_features", "detach_reference", "warmup_steps", "lambda_motion", "max_dt"},
    "replay": {"budget_mib", "ratio_schedule", "clip_len"},
    "supervision": {"pl_score_threshold", "conflict_iou", "pl_duplicate_iou"},
    "inference": {"score_threshold", "filter_threshold", "miss_tolerance", "duplicate_iou", "inference_dedup_enabled", "maximum_quantity"},
    "evaluation": {"manifest", "classes"},
    "report": {"redact_private_paths", "allow_estimated_metrics"},
}


def _validate_repair_v2(value: dict) -> None:
    if value.get("schema_version") != "cmot.repair_v2":
        raise ValueError("resolve_runtime_config requires cmot.repair_v2")
    for name, allowed in REPAIR_V2_NESTED_KEYS.items():
        section = value.get(name, {})
        if not isinstance(section, dict):
            raise ValueError("repair_v2 config section %s must be a mapping" % name)
        unknown = set(section) - allowed
        if unknown:
            raise ValueError("unknown keys in repair_v2 %s: %s" % (name, sorted(unknown)))
    if value.get("motion", {}).get("implementation") not in ("none", "one_step_residual_v2"):
        raise ValueError("repair_v2 motion implementation must be none or one_step_residual_v2")
    if value.get("motion", {}).get("mode") not in ("none", "one_step_agnostic_v2", "one_step_conditioned_v2"):
        raise ValueError("repair_v2 motion mode is not connected")


def resolve_runtime_config(curriculum: dict, runtime_paths: Mapping[str, Any], method: str, stage: str) -> dict:
    """Resolve one method/stage into the values consumed by train/infer/model."""
    _validate_repair_v2(curriculum)
    method = str(method)
    stage = str(stage)
    if method not in ("S0-repaired", "B1-repaired", "O1-residual-v2", "O2-residual-v2"):
        raise ValueError("unknown repair_v2 method %s" % method)
    resolved = {key: dict(value) if isinstance(value, dict) else value for key, value in curriculum.items()}
    resolved["runtime_paths"] = dict(runtime_paths)
    resolved["method"] = method
    resolved["stage"] = stage
    stage_ids = {
        "S0": {"active_global_ids": [206], "new_global_ids": [206], "old_global_ids": []},
        "S1": {"active_global_ids": [206, 792], "new_global_ids": [792], "old_global_ids": [206]},
        "S2": {"active_global_ids": [206, 792, 1122], "new_global_ids": [1122], "old_global_ids": [206, 792]},
    }
    if stage not in stage_ids:
        raise ValueError("repair_v2 stage must be S0, S1 or S2")
    resolved.update(stage_ids[stage])
    motion = dict(resolved["motion"])
    if method in ("S0-repaired", "B1-repaired"):
        motion["implementation"] = "none"
        motion["mode"] = "none"
    elif method == "O1-residual-v2":
        motion["implementation"] = "one_step_residual_v2"
        motion["mode"] = "one_step_agnostic_v2"
    else:
        motion["implementation"] = "one_step_residual_v2"
        motion["mode"] = "one_step_conditioned_v2"
    resolved["motion"] = motion
    resolved["actual_consumers"] = {
        "dataset": {"clip_frames": resolved["training"]["clip_frames"], "input_size": resolved["training"]["input_size"], "workers": resolved["training"]["workers"]},
        "optimizer": {key: resolved["training"][key] for key in ("lr_backbone", "lr_heads", "lr_motion", "weight_decay", "max_grad_norm")},
        "sampler": {"ratio_schedule": resolved["replay"]["ratio_schedule"], "seed": resolved["seed"]},
        "supervision": dict(resolved["supervision"]),
        "inference": dict(resolved["inference"]),
        "evaluation": dict(resolved["evaluation"]),
    }
    return resolved
