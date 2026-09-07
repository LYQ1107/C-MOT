"""Real-video optimizer runner for the repair_v2 protocol.

There is no smoke phase: the first successful optimizer update is step 1 of
the named run.  The checkpoint binds the model to the view, sampler and
resolved runtime configuration hashes used by inference and evaluation.
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .manifest import canonical_json_hash, sha256_file, write_json
from .ovtr_runtime import load_checkpoint, make_model


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "imgs": [value.to(device, non_blocking=True) for value in batch["imgs"]],
        "gt_instances": [value.to(device) for value in batch["gt_instances"]],
        "frame_metadata": batch.get("frame_metadata", []),
        "sample_metadata": batch.get("sample_metadata", {}),
    }


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _select_audit_parameters(model) -> Dict[str, torch.Tensor]:
    preferred = (
        "patch2query.weight",
        "bbox_embed.0.layers.0.weight",
        "track_embed.linear1.weight",
        "backbone.0.body.layer4.2.conv3.weight",
        "motion_head.net.1.weight",
    )
    all_params = dict(model.named_parameters())
    selected = {}
    for name in preferred:
        if name in all_params and all_params[name].requires_grad:
            selected[name] = all_params[name].detach().clone()
    for prefix in ("backbone.", "patch2query.", "bbox_embed.", "motion_head."):
        if any(name.startswith(prefix) for name in selected):
            continue
        for name, parameter in all_params.items():
            if name.startswith(prefix) and parameter.requires_grad:
                selected[name] = parameter.detach().clone()
                break
    return selected


def _gradient_audit(model, before: Dict[str, torch.Tensor]) -> dict:
    parameters = dict(model.named_parameters())
    result = {}
    for name, old in before.items():
        parameter = parameters.get(name)
        if parameter is None:
            continue
        grad = parameter.grad
        result[name] = {
            "grad_present": grad is not None,
            "grad_finite": bool(grad is not None and torch.isfinite(grad).all().item()),
            "grad_norm": None if grad is None else float(grad.detach().norm().item()),
            "update_norm": float((parameter.detach() - old).norm().item()),
        }
    return result


def _restore_rng(payload: Mapping[str, object]) -> None:
    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])


def _capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _optimizer(model, resolved: Mapping[str, object]):
    training = dict(resolved.get("training", {}))
    lr_backbone = float(training.get("lr_backbone", 2e-6))
    lr_heads = float(training.get("lr_heads", 2e-5))
    lr_motion = float(training.get("lr_motion", 2e-4))
    weight_decay = float(training.get("weight_decay", 1e-4))
    groups = {"backbone": [], "motion": [], "heads": []}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if name.startswith("backbone."):
            groups["backbone"].append(parameter)
        elif name.startswith("motion_head."):
            groups["motion"].append(parameter)
        else:
            groups["heads"].append(parameter)
    parameter_groups = []
    if groups["backbone"]:
        parameter_groups.append({"params": groups["backbone"], "lr": lr_backbone, "group_name": "backbone"})
    if groups["heads"]:
        parameter_groups.append({"params": groups["heads"], "lr": lr_heads, "group_name": "heads"})
    if groups["motion"]:
        parameter_groups.append({"params": groups["motion"], "lr": lr_motion, "group_name": "motion"})
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay), {
        "backbone": len(groups["backbone"]),
        "heads": len(groups["heads"]),
        "motion": len(groups["motion"]),
        "lr_backbone": lr_backbone,
        "lr_heads": lr_heads,
        "lr_motion": lr_motion,
        "weight_decay": weight_decay,
    }


def _exposure_update(exposure: dict, batch: dict) -> None:
    sample = batch.get("sample_metadata", {})
    stream = str(sample.get("stream", "current"))
    exposure["steps_by_stream"][stream] += 1
    exposure["clips_by_stream"][stream] += 1
    exposure["clip_ids"].add(str(sample.get("clip_id", "")))
    exposure["video_ids"].update(str(meta.get("video_id", "")) for meta in batch.get("frame_metadata", []))
    for meta in batch.get("frame_metadata", []):
        exposure["frames_seen"] += 1
        exposure["frame_keys"].add(str(meta.get("frame_key", "")))
    for target in batch.get("gt_instances", []):
        for index in range(len(target)):
            source = {0: "gt", 1: "gt_replay", 2: "pl"}.get(
                int(target.label_sources[index]) if target.has("label_sources") else 0,
                "unknown",
            )
            gid = str(int(target.labels[index]))
            exposure["annotations_by_source"][source] += 1
            exposure["annotations_by_global_id"][gid] += 1


def _finalize_exposure(exposure: dict) -> dict:
    result = dict(exposure)
    for key in ("clip_ids", "video_ids", "frame_keys"):
        result[key + "_unique"] = len(result.pop(key))
    result["steps_by_stream"] = dict(sorted(result["steps_by_stream"].items()))
    result["clips_by_stream"] = dict(sorted(result["clips_by_stream"].items()))
    result["annotations_by_source"] = dict(sorted(result["annotations_by_source"].items()))
    result["annotations_by_global_id"] = dict(
        sorted(result["annotations_by_global_id"].items(), key=lambda item: int(item[0]))
    )
    return result


def train(
    ovtr_root: str,
    config_file: str,
    text_embedding: str,
    image_embedding: Optional[str],
    train_view: str,
    image_root: str,
    output_dir: str,
    active_global_ids: Sequence[int],
    total_steps: int,
    checkpoint_init: str,
    device: str = "cuda",
    motion_mode: str = "none",
    label_mode: str = "complete",
    clip_len: int = 2,
    input_size: Sequence[int] = (640, 360),
    samples: int = 100000,
    resume: Optional[str] = None,
    alignment: bool = True,
    seed: int = 20260907,
    *,
    resolved_config: Optional[Mapping[str, object]] = None,
    replay_view: Optional[str] = None,
    experiment_id: Optional[str] = None,
    init_mode: Optional[str] = None,
    replay_memory_version: Optional[str] = None,
    protocol_hash: Optional[str] = None,
) -> dict:
    resolved = dict(resolved_config or {})
    training_cfg = dict(resolved.get("training", {}))
    replay_cfg = dict(resolved.get("replay", {}))
    if resolved:
        clip_len = int(training_cfg.get("clip_frames", clip_len))
        input_size = tuple(int(v) for v in training_cfg.get("input_size", input_size))
        motion_mode = str(dict(resolved.get("motion", {})).get("mode", motion_mode))
        label_mode = "partial" if str(resolved.get("stage", "")).startswith(("S1", "S2")) else label_mode
        seed = int(resolved.get("seed", seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(str(destination / "resolved_runtime_config.json"), resolved)

    model, criterion, registry, args, cfg = make_model(
        ovtr_root,
        config_file,
        text_embedding,
        image_embedding,
        active_global_ids,
        device,
        clip_len=clip_len,
        motion_mode=motion_mode,
        label_mode=label_mode,
        alignment=alignment,
        resolved_config=resolved or None,
    )
    from .data.real_video_dataset import ContinualVideoDataset, mot_collate_fn
    from .data.sampling import BalancedClipSampler

    checkpoint_payload = None
    if resume:
        checkpoint_payload = torch.load(resume, map_location="cpu")
        load_mode = "resume"
    else:
        load_mode = init_mode or "foundation"
    allowed_missing = ("motion_head.",) if load_mode == "stage_transfer" and motion_mode != "none" else ()
    init_audit = load_checkpoint(
        model,
        resume or checkpoint_init,
        init_mode=load_mode,
        strict=(load_mode == "resume"),
        allow_foundation_partial=(load_mode == "foundation"),
        allowed_missing_prefixes=allowed_missing,
    )
    optimizer, optimizer_groups = _optimizer(model, resolved)
    start_step = 0
    if checkpoint_payload is not None:
        if not checkpoint_payload.get("optimizer"):
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
        start_step = int(checkpoint_payload.get("step", 0))
        _restore_rng(checkpoint_payload.get("rng_state", {}))

    focus_ids = resolved.get("new_global_ids", [])
    if not isinstance(focus_ids, (list, tuple)):
        focus_ids = []
    dataset = ContinualVideoDataset(
        train_view,
        image_root,
        active_global_ids,
        clip_len=clip_len,
        input_size=(int(input_size[0]), int(input_size[1])),
        split="train",
        replay_path=replay_view,
        clip_strides=(1,),
        focus_global_ids=focus_ids,
    )
    sampler = BalancedClipSampler(
        dataset,
        total_steps=int(total_steps),
        start_step=start_step,
        seed=seed,
        stage_id=str(resolved.get("stage", destination.name)),
        stream_schedule=tuple(replay_cfg.get("ratio_schedule", ("current", "current", "current", "replay"))),
    )
    if checkpoint_payload is not None:
        sampler.load_state_dict(checkpoint_payload.get("sampler_state", {}))
    sampler_plan_hash = sampler.plan_hash()
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=int(training_cfg.get("workers", 0)),
        collate_fn=mot_collate_fn,
        pin_memory=False,
    )

    model.train()
    criterion.train()
    max_grad_norm = float(training_cfg.get("max_grad_norm", 0.1))
    step_log = destination / "train_steps.jsonl"
    log_mode = "a" if resume and step_log.exists() else "w"
    started = time.time()
    exposure = {
        "frames_seen": 0,
        "steps_by_stream": Counter(),
        "clips_by_stream": Counter(),
        "clip_ids": set(),
        "video_ids": set(),
        "frame_keys": set(),
        "annotations_by_source": Counter(),
        "annotations_by_global_id": Counter(),
    }
    runtime_stats = Counter()
    actual_steps = start_step
    with step_log.open(log_mode, encoding="utf-8") as log_handle:
        for step, raw_batch in enumerate(loader, start=start_step + 1):
            if step > int(total_steps):
                break
            batch = _move_batch(raw_batch, torch.device(device))
            _exposure_update(exposure, batch)
            optimizer.zero_grad(set_to_none=True)
            model.set_training_step(step)
            outputs = model(batch)
            loss_dict = criterion(outputs)
            weighted_terms = []
            raw_terms = {}
            for name, value in loss_dict.items():
                if name not in criterion.weight_dict:
                    continue
                if not _finite(value):
                    raise FloatingPointError("non-finite loss %s at step %d" % (name, step))
                raw_terms[name] = float(value.detach().item())
                weighted_terms.append(value * float(criterion.weight_dict[name]))
            if not weighted_terms:
                raise RuntimeError("no weighted losses reached the optimizer")
            loss = sum(weighted_terms)
            if not _finite(loss):
                raise FloatingPointError("non-finite total loss at step %d" % step)
            loss.backward()
            gradient_norm = float(clip_grad_norm_(model.parameters(), max_grad_norm).item())
            audit_before = _select_audit_parameters(model) if step in (1, int(total_steps)) else {}
            optimizer.step()
            parameter_audit = _gradient_audit(model, audit_before) if audit_before else {}
            step_runtime = model.consume_runtime_stats() if hasattr(model, "consume_runtime_stats") else {}
            runtime_stats.update({key: int(value) for key, value in step_runtime.items()})
            actual_steps = step
            record = {
                "step": step,
                "loss": float(loss.detach().item()),
                "gradient_norm_before_clip": gradient_norm,
                "weighted_loss_names": sorted(raw_terms),
                "losses": raw_terms,
                "parameter_audit": parameter_audit,
                "runtime_stats": step_runtime,
                "elapsed_s": round(time.time() - started, 3),
                "device": os.environ.get("CUDA_VISIBLE_DEVICES", device),
                "stream": raw_batch.get("sample_metadata", {}).get("stream", "current"),
            }
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
            log_handle.flush()

    if actual_steps != int(total_steps):
        raise RuntimeError("training ended at step %d, expected %d" % (actual_steps, int(total_steps)))
    exposure = _finalize_exposure(exposure)
    motion_cfg = dict(resolved.get("motion", {}))
    metadata = {
        "schema_version": "cmot.repair_v2.checkpoint",
        "experiment_id": experiment_id or destination.name,
        "stage": str(resolved.get("stage", destination.name)),
        "active_global_ids": list(active_global_ids),
        "motion_mode": motion_mode,
        "motion_implementation": motion_cfg.get(
            "implementation", "none" if motion_mode == "none" else "one_step_residual_v2"
        ),
        "label_mode": label_mode,
        "steps": int(actual_steps),
        "train_view_sha256": sha256_file(train_view),
        "replay_view_sha256": None if not replay_view else sha256_file(replay_view),
        "resolved_config_sha256": canonical_json_hash(resolved) if resolved else None,
        "sampler_plan_hash": sampler_plan_hash,
        "protocol_hash": protocol_hash,
        "replay_memory_version": replay_memory_version,
    }
    checkpoint_path = destination / ("checkpoint_%03d.pt" % int(actual_steps))
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(actual_steps),
        "stage": destination.name,
        "active_global_ids": list(active_global_ids),
        "motion_mode": motion_mode,
        "label_mode": label_mode,
        "cmot_metadata": metadata,
        "init_checkpoint": init_audit,
        "train_view": Path(train_view).name,
        "replay_view": None if not replay_view else Path(replay_view).name,
        "input_size": list(input_size),
        "sampler_state": sampler.state_dict(),
        "rng_state": _capture_rng(),
    }, str(checkpoint_path))
    summary = {
        "status": "OK",
        "scope": "full" if int(total_steps) >= 300 else "pilot",
        "stage": destination.name,
        "experiment_id": experiment_id or destination.name,
        "steps_requested": int(total_steps),
        "steps_completed": int(actual_steps),
        "start_step": int(start_step),
        "motion_mode": motion_mode,
        "motion_implementation": metadata["motion_implementation"],
        "label_mode": label_mode,
        "active_global_ids": list(active_global_ids),
        "train_view": Path(train_view).name,
        "replay_view": None if not replay_view else Path(replay_view).name,
        "train_view_sha256": metadata["train_view_sha256"],
        "replay_view_sha256": metadata["replay_view_sha256"],
        "checkpoint": Path(checkpoint_path).name,
        "checkpoint_sha256": sha256_file(str(checkpoint_path)),
        "checkpoint_metadata": metadata,
        "init_checkpoint": init_audit,
        "optimizer_groups": optimizer_groups,
        "sampler": {
            "dataset_clips": len(dataset),
            "dataset_stream_videos": len(dataset.stream_views),
            "plan_hash": sampler_plan_hash,
            "state": sampler.state_dict(),
        },
        "exposure": exposure,
        "runtime_stats": dict(sorted(runtime_stats.items())),
        "step_log": step_log.name,
        "elapsed_s": round(time.time() - started, 3),
    }
    write_json(str(destination / "train_summary.json"), summary)
    return summary


def _ids(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ovtr-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--text-embedding", required=True)
    parser.add_argument("--image-embedding")
    parser.add_argument("--train-view", required=True)
    parser.add_argument("--replay-view")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--active-global-ids", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--checkpoint-init", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--motion-mode", default="none")
    parser.add_argument("--label-mode", default="complete")
    parser.add_argument("--clip-len", type=int, default=2)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--no-alignment", action="store_true")
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    print(json.dumps(train(
        args.ovtr_root,
        args.config,
        args.text_embedding,
        args.image_embedding,
        args.train_view,
        args.image_root,
        args.output_dir,
        _ids(args.active_global_ids),
        args.total_steps,
        args.checkpoint_init,
        args.device,
        args.motion_mode,
        args.label_mode,
        args.clip_len,
        (args.input_width, args.input_height),
        args.samples,
        args.resume,
        not args.no_alignment,
        args.seed,
        replay_view=args.replay_view,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
