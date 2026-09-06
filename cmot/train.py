"""Real-video optimizer runner used for the C-MOT pilot/full step counts."""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .manifest import sha256_file, write_json
from .ovtr_runtime import load_checkpoint, make_model


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "imgs": [value.to(device, non_blocking=True) for value in batch["imgs"]],
        "gt_instances": [value.to(device) for value in batch["gt_instances"]],
        "frame_metadata": batch.get("frame_metadata", []),
    }


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _gradient_audit(model, before: Dict[str, torch.Tensor]) -> dict:
    result = {}
    for name, old in before.items():
        param = dict(model.named_parameters()).get(name)
        if param is None:
            continue
        grad = param.grad
        result[name] = {
            "grad_present": grad is not None,
            "grad_finite": bool(grad is not None and torch.isfinite(grad).all().item()),
            "grad_norm": None if grad is None else float(grad.detach().norm().item()),
            "update_norm": float((param.detach() - old).norm().item()),
        }
    return result


def _select_audit_parameters(model) -> Dict[str, torch.Tensor]:
    names = (
        "patch2query.weight",
        "bbox_embed.0.layers.0.weight",
        "track_embed.linear1.weight",
        "backbone.0.body.layer4.2.conv3.weight",
        "motion_head.net.1.weight",
    )
    all_params = dict(model.named_parameters())
    return {name: all_params[name].detach().clone() for name in names if name in all_params and all_params[name].requires_grad}


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
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
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
    )
    # make_model() must run first so OVTR's bundled detectron2 structures are
    # selected before the real-video adapter imports Instances.
    from .data.real_video_dataset import ContinualVideoDataset, mot_collate_fn

    init_audit = load_checkpoint(model, resume or checkpoint_init, strict=bool(resume), allow_foundation_partial=not bool(resume))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-5,
        weight_decay=1e-4,
    )
    start_step = 0
    if resume:
        payload = torch.load(resume, map_location="cpu")
        if payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
    dataset = ContinualVideoDataset(
        train_view,
        image_root,
        active_global_ids,
        clip_len=clip_len,
        input_size=(int(input_size[0]), int(input_size[1])),
        samples=samples,
        split="train",
    )
    model.train()
    criterion.train()
    step_log = destination / "train_steps.jsonl"
    log_mode = "a" if resume and step_log.exists() else "w"
    audit_parameters = _select_audit_parameters(model)
    last_checkpoint = None
    started = time.time()
    with step_log.open(log_mode, encoding="utf-8") as log_handle:
        for step in range(start_step + 1, int(total_steps) + 1):
            batch = _move_batch(dataset[(step - 1) % len(dataset)], torch.device(device))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss_dict = criterion(outputs)
            weighted_terms = []
            raw_terms = {}
            for name, value in loss_dict.items():
                if name not in criterion.weight_dict:
                    continue
                raw_terms[name] = float(value.detach().item())
                weighted_terms.append(value * float(criterion.weight_dict[name]))
            if not weighted_terms:
                raise RuntimeError("no weighted losses reached the optimizer")
            loss = sum(weighted_terms)
            if not _finite(loss):
                raise FloatingPointError("non-finite total loss at step %d" % step)
            loss.backward()
            gradient_norm = float(clip_grad_norm_(model.parameters(), 0.1).item())
            optimizer.step()
            parameter_audit = _gradient_audit(model, audit_parameters)
            audit_parameters = _select_audit_parameters(model)
            record = {
                "step": step,
                "loss": float(loss.detach().item()),
                "gradient_norm_before_clip": gradient_norm,
                "weighted_loss_names": sorted(raw_terms),
                "losses": raw_terms,
                "parameter_audit": parameter_audit,
                "elapsed_s": round(time.time() - started, 3),
                "device": os.environ.get("CUDA_VISIBLE_DEVICES", device),
            }
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
            log_handle.flush()
            if step in (20, 100, 300) or step == int(total_steps):
                checkpoint_path = destination / ("checkpoint_%03d.pt" % step)
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "stage": destination.name,
                    "active_global_ids": list(active_global_ids),
                    "motion_mode": motion_mode,
                    "label_mode": label_mode,
                    "init_checkpoint": init_audit,
                    "train_view": Path(train_view).name,
                    "input_size": list(input_size),
                }, str(checkpoint_path))
                last_checkpoint = str(checkpoint_path)
    summary = {
        "status": "OK",
        "stage": destination.name,
        "steps_completed": int(total_steps),
        "start_step": start_step,
        "motion_mode": motion_mode,
        "label_mode": label_mode,
        "active_global_ids": list(active_global_ids),
        "train_view": Path(train_view).name,
        "checkpoint": None if last_checkpoint is None else Path(last_checkpoint).name,
        "checkpoint_sha256": None if last_checkpoint is None else sha256_file(last_checkpoint),
        "init_checkpoint": init_audit,
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
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
