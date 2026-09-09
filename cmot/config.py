"""Strict loaders and runtime resolution for C-MOT curricula."""

from pathlib import Path
from typing import Any, Mapping

import yaml


ALLOWED_KEYS = {
    "project", "schema_version", "seed", "network", "resources", "protocol",
    "training", "augmentation", "motion", "replay", "report", "supervision", "pseudo",
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

# V4 deliberately has its own schema contract.  Keep the V3 key set intact so
# old curricula continue to reject V4-only knobs rather than silently ignoring
# them.
V4_NESTED_KEYS = {name: set(keys) for name, keys in V3_NESTED_KEYS.items()}
V4_NESTED_KEYS["motion"].update({
    "history_length", "history_hidden_dim", "num_modes", "semantic_dim",
    "min_history_points", "detach_history", "reference_enabled",
})

# COOLer compatibility is intentionally a separate contract.  In particular,
# it must not inherit V3/V4 replay, calibration, motion, or pilot caps through
# permissive defaults.  Keep the key set explicit so a typo cannot silently
# change the protocol.
COOLER_NESTED_KEYS = {
    "network": {"require_route_audit", "fallback_to_proxy"},
    "resources": {"protected_gpu_ids", "max_gpus", "dataloader_workers_per_trainer"},
    "protocol": {
        "name", "class_order", "replay_free", "previous_training_data_forbidden",
        "full_train_split", "full_validation_split", "expected_train_videos",
        "expected_val_videos", "train_source_cap", "eval_video_cap", "train_split",
        "eval_split", "reference_scope", "num_ref_imgs", "skip_nomatch_samples",
        "nominal_epoch_length", "new_gt_exclusion_iou",
    },
    "training": {
        "optimizer", "batch_size_per_gpu", "gradient_accumulation_steps", "epochs",
        "workers", "input_size", "lr_backbone", "lr_heads", "weight_decay",
        "max_grad_norm", "lr_decay_milestones", "lr_decay_gamma",
    },
    "augmentation": {"horizontal_flip_probability", "pair_consistent_flip"},
    "motion": {"enabled", "implementation", "mode"},
    "supervision": {"lambda_gt", "lambda_pl", "pl_warmup_steps", "label_mode"},
    "pseudo": {
        "policy_version", "old_gt_calibration", "exclusion_iou_with_new_gt",
        "total_segment_cap", "total_frame_cap", "per_class_segment_cap",
        "min_segment_frames",
    },
    "distillation": {"enabled"},
    "inference": {
        "birth_threshold", "keep_threshold", "export_threshold", "miss_tolerance",
        "duplicate_iou", "duplicate_feature_cos", "maximum_quantity",
        "inference_dedup_enabled",
    },
    "evaluation": {"classes", "hota_thresholds", "aggregation", "full_validation"},
    "report": {"redact_private_paths", "allow_estimated_metrics", "require_raw_metrics"},
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


def _validate_v4(value: dict) -> None:
    _validate_nested(value, V4_NESTED_KEYS, "continual_v4")
    motion = value.get("motion", {})
    if motion.get("implementation", "none") not in ("none", "history_gru_v1"):
        raise ValueError("continual_v4 motion implementation is not connected")
    if motion.get("mode", "none") not in (
        "none", "history_agnostic_v1", "history_conditioned_v1"
    ):
        raise ValueError("continual_v4 motion mode is not connected")
    if int(motion.get("num_modes", 1)) != 1:
        raise ValueError("continual_v4 requires num_modes=1")
    if int(motion.get("min_history_points", 2)) < 1:
        raise ValueError("continual_v4 min_history_points must be at least one")
    inference = value.get("inference", {})
    required = ("birth_threshold", "keep_threshold", "export_threshold", "duplicate_feature_cos")
    missing = [key for key in required if key not in inference]
    if missing:
        raise ValueError("continual_v4 inference contract missing: %s" % ",".join(missing))
    if not 0 <= float(inference["birth_threshold"]) <= 1 or not 0 <= float(inference["export_threshold"]) <= 1:
        raise ValueError("continual_v4 inference thresholds must be probabilities")


def _validate_cooler(value: dict) -> None:
    _validate_nested(value, COOLER_NESTED_KEYS, "cooler_compat")
    protocol = value.get("protocol", {})
    expected_order = [["car"], ["pedestrian"], ["truck"]]
    if protocol.get("class_order") != expected_order:
        raise ValueError("cooler_compat class_order must be car -> pedestrian -> truck")
    for key in ("replay_free", "previous_training_data_forbidden", "full_train_split", "full_validation_split"):
        if protocol.get(key) is not True:
            raise ValueError("cooler_compat protocol.%s must be true" % key)
    if int(protocol.get("expected_train_videos", 0)) != 1400 or int(protocol.get("expected_val_videos", 0)) != 200:
        raise ValueError("cooler_compat requires expected BDD counts 1400/200")
    if protocol.get("train_source_cap") is not None or protocol.get("eval_video_cap") is not None:
        raise ValueError("cooler_compat does not permit train/eval caps")
    if int(protocol.get("reference_scope", 0)) != 3 or int(protocol.get("num_ref_imgs", 0)) != 1:
        raise ValueError("cooler_compat reference sampler must be scope=3 and num_ref_imgs=1")
    if protocol.get("skip_nomatch_samples") is not True:
        raise ValueError("cooler_compat requires skip_nomatch_samples")
    training = value.get("training", {})
    if str(training.get("optimizer", "")) != "AdamW":
        raise ValueError("cooler_compat uses the architecture-native AdamW optimizer")
    if int(training.get("epochs", 0)) != 6 or int(training.get("gradient_accumulation_steps", 0)) != 16:
        raise ValueError("cooler_compat requires 6 epochs and accumulation=16")
    if [int(v) for v in training.get("input_size", [])] != [1280, 720]:
        raise ValueError("cooler_compat input_size must be [1280, 720] (width,height)")
    if [int(v) for v in training.get("lr_decay_milestones", [])] != [4, 5]:
        raise ValueError("cooler_compat LR milestones must be [4,5]")
    augmentation = value.get("augmentation", {})
    if abs(float(augmentation.get("horizontal_flip_probability", -1.0)) - 0.5) > 1e-9:
        raise ValueError("cooler_compat horizontal flip probability must be 0.5")
    if augmentation.get("pair_consistent_flip") is not True:
        raise ValueError("cooler_compat requires pair-consistent flip")
    motion = value.get("motion", {})
    if motion.get("enabled", False) or motion.get("implementation", "none") != "none" or motion.get("mode", "none") != "none":
        raise ValueError("cooler_compat motion must be disabled")
    supervision = value.get("supervision", {})
    if str(supervision.get("label_mode", "")) != "cooler_complete_seen":
        raise ValueError("cooler_compat requires cooler_complete_seen labels")
    if float(supervision.get("lambda_gt", -1.0)) != 1.0 or float(supervision.get("lambda_pl", -1.0)) != 1.0:
        raise ValueError("cooler_compat GT/PL weights must both be 1.0")
    if int(supervision.get("pl_warmup_steps", -1)) != 0:
        raise ValueError("cooler_compat PL warmup must be zero")
    pseudo = value.get("pseudo", {})
    if str(pseudo.get("policy_version", "")) != "cooler_track_pl_v1":
        raise ValueError("cooler_compat pseudo policy must be cooler_track_pl_v1")
    if pseudo.get("old_gt_calibration") is not False:
        raise ValueError("cooler_compat forbids old-GT calibration")
    for key in ("total_segment_cap", "total_frame_cap", "per_class_segment_cap"):
        if pseudo.get(key) is not None:
            raise ValueError("cooler_compat pseudo.%s must be null" % key)
    if value.get("distillation", {}).get("enabled") is not False:
        raise ValueError("cooler_compat distillation must be disabled")
    inference = value.get("inference", {})
    required = ("birth_threshold", "keep_threshold", "export_threshold", "miss_tolerance", "duplicate_iou", "duplicate_feature_cos")
    missing = [key for key in required if key not in inference]
    if missing:
        raise ValueError("cooler_compat inference contract missing: %s" % ",".join(missing))
    for key in ("birth_threshold", "keep_threshold", "export_threshold", "duplicate_iou", "duplicate_feature_cos"):
        if not 0.0 <= float(inference[key]) <= 1.0:
            raise ValueError("cooler_compat inference.%s must be a probability" % key)
    if value.get("report", {}).get("allow_estimated_metrics") is not False:
        raise ValueError("cooler_compat reports cannot use estimated metrics")


def _cooler_stage_ids(stage: str) -> dict:
    stages = {
        "s0": {"active_names": ["car"], "new_names": ["car"], "old_names": []},
        "s1": {"active_names": ["car", "pedestrian"], "new_names": ["pedestrian"], "old_names": ["car"]},
        "s2": {"active_names": ["car", "pedestrian", "truck"], "new_names": ["truck"], "old_names": ["car", "pedestrian"]},
        "oracle_s2": {"active_names": ["car", "pedestrian", "truck"], "new_names": ["car", "pedestrian", "truck"], "old_names": []},
    }
    key = str(stage).lower()
    if key not in stages:
        raise ValueError("unknown cooler stage %s" % stage)
    return dict(stages[key])


def resolve_cooler_runtime_config(curriculum: dict, runtime_paths: Mapping[str, Any], stage: str) -> dict:
    """Materialize the independent COOLer-compatible runtime contract."""
    _validate_cooler(curriculum)
    from .class_registry import tao_bdd_registry

    result = _copy_sections(curriculum)
    stage_ids = _cooler_stage_ids(stage)
    registry = tao_bdd_registry()
    active_ids = [registry.classes[name].global_semantic_id for name in stage_ids["active_names"]]
    new_ids = [registry.classes[name].global_semantic_id for name in stage_ids["new_names"]]
    old_ids = [registry.classes[name].global_semantic_id for name in stage_ids["old_names"]]
    stage_key = str(stage).lower()
    method_names = {
        "s0": "CMOT-COOLER-S0",
        "s1": "CMOT-COOLER-S1",
        "s2": "CMOT-COOLER-S2",
        "oracle_s2": "CMOT-ORACLE-S2",
    }
    result.update({
        "runtime_paths": dict(runtime_paths),
        "stage": stage_key,
        "method": method_names[stage_key],
        "protocol_role": "oracle" if stage_key == "oracle_s2" else "cooler_cil",
        "active_global_ids": active_ids,
        "new_global_ids": new_ids,
        "old_global_ids": old_ids,
        "label_mode": "cooler_complete_seen",
        "enable_pl": stage_key in ("s1", "s2"),
        "enable_kd": False,
        "enable_motion": False,
    })
    training = dict(result.get("training", {}))
    training["input_size"] = [1280, 720]
    result["training"] = training
    supervision = dict(result.get("supervision", {}))
    supervision.update({"lambda_gt": 1.0, "lambda_pl": 1.0, "pl_warmup_steps": 0, "label_mode": "cooler_complete_seen"})
    result["supervision"] = supervision
    motion = dict(result.get("motion", {}))
    motion.update({"enabled": False, "implementation": "none", "mode": "none"})
    result["motion"] = motion
    distillation = dict(result.get("distillation", {}))
    distillation["enabled"] = False
    result["distillation"] = distillation
    inference = dict(result.get("inference", {}))
    inference.setdefault("maximum_quantity", 160)
    inference.setdefault("inference_dedup_enabled", True)
    result["inference"] = inference
    result["actual_consumers"] = {
        "schema_version": "cmot.cooler_compat.runtime.v1",
        "stage": result["stage"],
        "active_global_ids": active_ids,
        "new_global_ids": new_ids,
        "old_global_ids": old_ids,
        "label_mode": result["label_mode"],
        "motion": {"enabled": False, "implementation": "none", "mode": "none"},
        "replay": {"enabled": False, "gt_replay": 0},
        "distillation": {"enabled": False},
        "training": {
            "epochs": int(training["epochs"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "input_size": list(training["input_size"]),
            "lr_decay_milestones": list(training["lr_decay_milestones"]),
        },
        "sampling": {
            "reference_scope": int(result["protocol"]["reference_scope"]),
            "num_ref_imgs": int(result["protocol"]["num_ref_imgs"]),
            "skip_nomatch_samples": True,
            "seed": int(result.get("seed", 777)),
        },
    }
    return result


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
    elif schema == "cmot.continual_v4":
        _validate_v4(value)
    elif schema == "cmot.cooler_compat.v1":
        _validate_cooler(value)
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
        raise ValueError("continual stage must be S0, J3, S1 or S2")
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
        "V4-B0": (True, False, False),
        "V4-HIST-AGN": (True, False, True),
        "V4-HIST-COND": (True, False, True),
        "V4-HIST-COND-AUX": (True, False, True),
    }
    if method not in methods:
        raise ValueError("unknown continual method %s" % method)
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
    elif schema == "cmot.continual_v4":
        _validate_v4(curriculum)
    elif schema == "cmot.cooler_compat.v1":
        return resolve_cooler_runtime_config(curriculum, runtime_paths, stage)
    else:
        raise ValueError("resolve_runtime_config requires a V2, V3 or V4 curriculum")
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
    elif str(method) == "O1-residual-v2":
        motion["implementation"] = "one_step_residual_v2"
        motion["mode"] = "one_step_agnostic_v2"
    elif str(method) == "O2-residual-v2":
        motion["implementation"] = "one_step_residual_v2"
        motion["mode"] = "one_step_conditioned_v2"
    elif str(method) == "V4-HIST-AGN":
        motion["implementation"] = "history_gru_v1"
        motion["mode"] = "history_agnostic_v1"
        motion["reference_enabled"] = True
    elif str(method) in ("V4-HIST-COND", "V4-HIST-COND-AUX"):
        motion["implementation"] = "history_gru_v1"
        motion["mode"] = "history_conditioned_v1"
        motion["reference_enabled"] = str(method) != "V4-HIST-COND-AUX"
    elif str(method) == "V4-B0":
        motion["implementation"] = "none"
        motion["mode"] = "none"
    else:
        raise ValueError("motion-enabled method has no explicit mapping: %s" % method)
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
