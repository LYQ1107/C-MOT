"""Strict loader for the public C-MOT curriculum configuration."""

from pathlib import Path
from typing import Any, Mapping

import yaml


ALLOWED_KEYS = {
    "project", "schema_version", "seed", "network", "resources", "protocol",
    "training", "motion", "replay", "report",
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
    if value.get("schema_version") != "cmot.curriculum.v1":
        raise ValueError("unsupported curriculum schema_version")
    for name, allowed in NESTED_KEYS.items():
        section = value.get(name, {})
        if not isinstance(section, dict):
            raise ValueError("config section %s must be a mapping" % name)
        unknown_nested = set(section) - allowed
        if unknown_nested:
            raise ValueError("unknown keys in %s: %s" % (name, sorted(unknown_nested)))
    return value
