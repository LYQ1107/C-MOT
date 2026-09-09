"""Six-epoch, replay-free trainer for the COOLer-compatible protocol."""

import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.cooler_protocol import CoolerCompatiblePairDataset
from .manifest import canonical_json_hash, sha256_file, write_json
from .ovtr_runtime import load_checkpoint, make_model
from .train import (
    _clip_optimizer_groups,
    _finite,
    _gradient_audit,
    _move_batch,
    _optimizer,
    _select_audit_parameters,
)


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _exposure_update(exposure: dict, batch: Mapping[str, Any], old_ids: Sequence[int]) -> None:
    sample = dict(batch.get("sample_metadata", {}))
    exposure["micro_batches_seen"] += 1
    exposure["effective_pairs_seen"] += 1
    exposure["video_uids"].add(str(sample.get("source_video_uid", sample.get("video_id", ""))))
    exposure["fallback_samples"] += int(bool(sample.get("fallback_used", False)))
    exposure["flip_pairs"] += int(bool(sample.get("pair_consistent_flip", False)))
    exposure["frames_seen"] += len(batch.get("frame_metadata", []))
    for meta in batch.get("frame_metadata", []):
        exposure["frame_keys"].add(str(meta.get("frame_key", "")))
    for target in batch.get("gt_instances", []):
        for index in range(len(target)):
            source = {0: "gt", 1: "gt_replay", 2: "pl"}.get(
                int(target.label_sources[index]) if target.has("label_sources") else 0,
                "unknown",
            )
            gid = int(target.labels[index])
            name = {206: "car", 792: "pedestrian", 1122: "truck"}.get(gid, str(gid))
            exposure["annotations_by_source"][source] += 1
            exposure["annotations_by_class"][name] += 1
            if source == "gt":
                exposure["gt_boxes"] += 1
                exposure["gt_tracks"].add((str(sample.get("source_video_uid", "")), int(target.obj_ids[index])))
            elif source == "pl":
                exposure["pseudo_boxes"] += 1
                exposure["pseudo_tracks"].add((str(sample.get("source_video_uid", "")), int(target.obj_ids[index])))
            if source == "gt_replay":
                exposure["replay_boxes"] += 1
            if source == "gt" and gid in {int(value) for value in old_ids}:
                exposure["old_real_gt_boxes"] += 1


def _new_exposure() -> dict:
    return {
        "micro_batches_seen": 0,
        "optimizer_steps": 0,
        "effective_pairs_seen": 0,
        "frames_seen": 0,
        "video_uids": set(),
        "frame_keys": set(),
        "annotations_by_source": Counter(),
        "annotations_by_class": Counter(),
        "gt_boxes": 0,
        "gt_tracks": set(),
        "pseudo_boxes": 0,
        "pseudo_tracks": set(),
        "replay_boxes": 0,
        "old_real_gt_boxes": 0,
        "fallback_samples": 0,
        "flip_pairs": 0,
    }


def _finish_exposure(exposure: dict) -> dict:
    result = dict(exposure)
    for key in ("video_uids", "frame_keys", "gt_tracks", "pseudo_tracks"):
        result[key + "_unique"] = len(result.pop(key))
    result["annotations_by_source"] = dict(sorted(result["annotations_by_source"].items()))
    result["annotations_by_class"] = dict(sorted(result["annotations_by_class"].items()))
    result["gt_replay"] = int(result["replay_boxes"])
    return result


def _weighted_losses(criterion, outputs) -> tuple[torch.Tensor, dict]:
    loss_dict = criterion(outputs)
    weighted = []
    raw = {}
    for name, value in loss_dict.items():
        if name not in criterion.weight_dict:
            raise RuntimeError("loss %s is absent from criterion.weight_dict" % name)
        if not _finite(value):
            raise FloatingPointError("non-finite loss %s" % name)
        raw[name] = float(value.detach().item())
        weighted.append(value * float(criterion.weight_dict[name]))
    if not weighted:
        raise RuntimeError("no weighted losses reached the optimizer")
    loss = sum(weighted)
    if not _finite(loss):
        raise FloatingPointError("non-finite total loss")
    return loss, raw


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    metadata: Mapping[str, Any],
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "stage": str(metadata["stage"]),
        "active_global_ids": list(metadata["active_global_ids"]),
        "motion_mode": "none",
        "label_mode": "cooler_complete_seen",
        "cmot_metadata": dict(metadata),
    }


def train_cooler_stage(
    *,
    ovtr_root: str,
    config_file: str,
    text_embedding: str,
    image_embedding: Optional[str],
    train_view: str,
    image_root: str,
    output_dir: str,
    active_global_ids: Sequence[int],
    old_global_ids: Sequence[int],
    parent_checkpoint: str,
    parent_init_mode: str,
    resolved_config: Mapping[str, Any],
    pseudo_manifest: Optional[str] = None,
    device: str = "cuda",
    resume: Optional[str] = None,
) -> dict:
    """Train one protocol stage and save epoch_01.pt through epoch_06.pt."""
    cfg = dict(resolved_config)
    training_cfg = dict(cfg.get("training", {}))
    protocol = dict(cfg.get("protocol", {}))
    if cfg.get("label_mode") != "cooler_complete_seen":
        raise ValueError("COOLer trainer requires cooler_complete_seen")
    if cfg.get("enable_motion") or cfg.get("enable_kd"):
        raise ValueError("COOLer trainer is motion/KD-free")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _seed_all(int(cfg.get("seed", 777)))
    model, criterion, _, _, _ = make_model(
        ovtr_root,
        config_file,
        text_embedding,
        image_embedding,
        active_global_ids,
        device,
        clip_len=2,
        motion_mode="none",
        label_mode="cooler_complete_seen",
        alignment=bool(image_embedding),
        resolved_config=cfg,
    )
    # Import the dataset helper only after make_model has installed OVTR's
    # bundled detectron2 compatibility modules on sys.path.  This keeps
    # config/audit commands independent of the training runtime import order.
    from .data.real_video_dataset import mot_collate_fn

    init_audit = load_checkpoint(
        model,
        parent_checkpoint,
        init_mode=parent_init_mode,
        allow_foundation_partial=parent_init_mode == "foundation",
    )
    optimizer, optimizer_groups = _optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(value) for value in training_cfg.get("lr_decay_milestones", [4, 5])],
        gamma=float(training_cfg.get("lr_decay_gamma", 0.1)),
    )
    start_epoch = 0
    optimizer_steps = 0
    if resume:
        payload = torch.load(resume, map_location="cpu")
        load_checkpoint(model, resume, init_mode="resume")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", 0))
        optimizer_steps = int(payload.get("step", 0))
    dataset = CoolerCompatiblePairDataset(
        train_view,
        image_root,
        active_global_ids,
        reference_scope=int(protocol.get("reference_scope", 3)),
        horizontal_flip_probability=float(cfg.get("augmentation", {}).get("horizontal_flip_probability", 0.5)),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(training_cfg.get("workers", 0)),
        collate_fn=mot_collate_fn,
        pin_memory=False,
    )
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 16))
    epochs = int(training_cfg.get("epochs", 6))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 0.1))
    exposure = _new_exposure()
    step_records = []
    epoch_records = []
    clip_records = []
    parameter_audits = {}
    runtime_stats = Counter()
    last_checkpoint_metadata = {}
    started = time.time()
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        plan_hash = dataset.plan_hash()
        clip_records.append({"epoch": epoch + 1, "plan_sha256": plan_hash, "nominal_length": len(dataset)})
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_micro = 0
        epoch_optimizer_steps = 0
        for micro_index, raw_batch in enumerate(loader):
            batch = _move_batch(raw_batch, torch.device(device))
            _exposure_update(exposure, batch, old_global_ids)
            epoch_micro += 1
            window_start = (micro_index // accumulation) * accumulation
            actual_window_count = min(accumulation, len(dataset) - window_start)
            model.set_training_step(optimizer_steps + 1)
            outputs = model(batch)
            loss, raw_losses = _weighted_losses(criterion, outputs)
            (loss / float(actual_window_count)).backward()
            if (epoch_micro % accumulation == 0) or (micro_index + 1 == len(dataset)):
                nonfinite = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
                ]
                if nonfinite:
                    raise FloatingPointError("non-finite gradients: %s" % nonfinite[:8])
                audit_before = _select_audit_parameters(model) if optimizer_steps == 0 else {}
                clipping = _clip_optimizer_groups(optimizer, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                epoch_optimizer_steps += 1
                exposure["optimizer_steps"] = optimizer_steps
                if audit_before:
                    parameter_audits["step_1"] = _gradient_audit(model, audit_before)
                step_runtime = model.consume_runtime_stats() if hasattr(model, "consume_runtime_stats") else {}
                runtime_stats.update({key: int(value) for key, value in step_runtime.items()})
                step_records.append({
                    "epoch": epoch + 1,
                    "optimizer_step": optimizer_steps,
                    "micro_batches_seen": exposure["micro_batches_seen"],
                    "loss": float(loss.detach().item()),
                    "losses": raw_losses,
                    "actual_accumulation_count": actual_window_count,
                    "gradient_clipping": clipping,
                    "runtime_stats": step_runtime,
                })
        scheduler.step()
        epoch_records.append({
            "epoch": epoch + 1,
            "nominal_train_frames": len(dataset),
            "micro_batches_seen": epoch_micro,
            "optimizer_steps": epoch_optimizer_steps,
            "current_lr": [float(group["lr"]) for group in optimizer.param_groups],
            "sampling_plan_sha256": plan_hash,
        })
        metadata = {
            "schema_version": "cmot.cooler_compat.checkpoint.v1",
            "stage": str(cfg["stage"]),
            "method": str(cfg["method"]),
            "epoch": int(epoch + 1),
            "optimizer_steps": int(optimizer_steps),
            "micro_batches": int(exposure["micro_batches_seen"]),
            "effective_pairs_seen": int(exposure["effective_pairs_seen"]),
            "active_global_ids": [int(value) for value in active_global_ids],
            "old_global_ids": [int(value) for value in old_global_ids],
            "motion_mode": "none",
            "label_mode": "cooler_complete_seen",
            "train_view_sha256": sha256_file(train_view),
            "pseudo_manifest_sha256": None if not pseudo_manifest else sha256_file(pseudo_manifest),
            "sampling_plan_sha256": plan_hash,
            "resolved_config_sha256": canonical_json_hash(cfg),
            "parent_checkpoint_sha256": init_audit["sha256"],
            "foundation_checkpoint_sha256": init_audit["sha256"] if parent_init_mode == "foundation" else None,
            "current_lr": [float(group["lr"]) for group in optimizer.param_groups],
            "replay_free": True,
            "distillation_enabled": False,
        }
        last_checkpoint_metadata = dict(metadata)
        torch.save(
            _checkpoint_payload(model, optimizer, scheduler, optimizer_steps, epoch + 1, metadata),
            str(destination / ("epoch_%02d.pt" % (epoch + 1))),
        )
    if start_epoch >= epochs:
        final_epoch = epochs
    else:
        final_epoch = epochs
    if len(epoch_records) != max(0, epochs - start_epoch):
        raise RuntimeError("COOLer trainer did not complete all requested epochs")
    finished_exposure = _finish_exposure(exposure)
    if finished_exposure.get("gt_replay", 0) != 0:
        raise RuntimeError("COOLer replay-free audit failed: gt_replay is non-zero")
    if finished_exposure.get("old_real_gt_boxes", 0) != 0:
        raise RuntimeError("COOLer stage audit failed: old real GT reached training")
    final_checkpoint = destination / ("epoch_%02d.pt" % final_epoch)
    final_sha = sha256_file(str(final_checkpoint))
    plan_sha = canonical_json_hash(clip_records)
    summary = {
        "status": "OK",
        "scope": "full",
        "schema_version": "cmot.cooler_compat.training_summary.v1",
        "stage": cfg["stage"],
        "method": cfg["method"],
        "epochs_requested": epochs,
        "epochs_completed": final_epoch,
        "micro_batches_seen": int(exposure["micro_batches_seen"]),
        "optimizer_steps": int(optimizer_steps),
        "effective_pairs_seen": int(exposure["effective_pairs_seen"]),
        "effective_batch": int(accumulation),
        "nominal_train_frames": int(len(dataset)),
        "train_videos": int(len(dataset.videos)),
        "sampling_plan_sha256": plan_sha,
        "sampling_epochs": clip_records,
        "epoch_records": epoch_records,
        "checkpoint": final_checkpoint.name,
        "checkpoint_sha256": final_sha,
        "checkpoint_metadata": last_checkpoint_metadata,
        "parent_checkpoint_sha256": init_audit["sha256"],
        "init_checkpoint": init_audit,
        "optimizer_groups": optimizer_groups,
        "exposure": finished_exposure,
        "gradient_audit": parameter_audits,
        "runtime_stats": dict(sorted(runtime_stats.items())),
        "train_steps": step_records,
        "elapsed_s": round(time.time() - started, 3),
        "hardware_not_matched": True,
    }
    write_json(str(destination / "training_audit.json"), summary)
    return summary


__all__ = ["train_cooler_stage"]
