"""CLI for asset checks and raw real-video prediction generation."""

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .manifest import canonical_json_hash, write_json
from .ovtr_runtime import load_checkpoint, make_model, run_video_inference


def _ids(value: str):
    return [int(item) for item in value.split(",") if item]


def infer_checkpoint(
    ovtr_root: str,
    config_file: str,
    text_embedding: str,
    image_embedding: Optional[str],
    checkpoint: str,
    view: str,
    image_root: str,
    output_jsonl: str,
    output_summary: str,
    active_global_ids: Sequence[int],
    resolved_config: Optional[Mapping] = None,
    device: str = "cuda",
    split: str = "val",
    video_ids: Optional[Sequence[str]] = None,
    alignment: bool = True,
    checkpoint_role: str = "trained",
) -> dict:
    """Bind one checkpoint, one immutable view and one prediction artifact."""
    resolved = dict(resolved_config or {})
    inference_cfg = dict(resolved.get("inference", {}))
    model, _, _, _, _ = make_model(
        ovtr_root,
        config_file,
        text_embedding,
        image_embedding,
        active_global_ids,
        device,
        clip_len=int(dict(resolved.get("training", {})).get("clip_frames", 4)),
        motion_mode=str(dict(resolved.get("motion", {})).get("mode", "none")),
        label_mode="partial" if str(resolved.get("stage", "")).startswith(("S1", "S2")) else "complete",
        score_threshold=float(inference_cfg.get("score_threshold", 0.19)),
        filter_threshold=float(inference_cfg.get("filter_threshold", 0.19)),
        miss_tolerance=int(inference_cfg.get("miss_tolerance", 5)),
        maximum_quantity=int(inference_cfg.get("maximum_quantity", 160)),
        alignment=alignment,
        resolved_config=resolved or None,
    )
    audit = load_checkpoint(
        model,
        checkpoint,
        init_mode="resume",
        expected_metadata={
            "active_global_ids": list(active_global_ids),
            "motion_mode": str(dict(resolved.get("motion", {})).get("mode", "none")),
        } if resolved else None,
    )
    view_manifest_hash = json.loads(Path(view).read_text(encoding="utf-8")).get("manifest_hash")
    result = run_video_inference(
        model,
        view,
        image_root,
        output_jsonl,
        split,
        video_ids=video_ids,
        input_size=tuple(dict(resolved.get("training", {})).get("input_size", [640, 360])),
        metadata={
            "checkpoint_sha256": audit["sha256"],
            "teacher_checkpoint_sha256": audit["sha256"],
            "active_global_ids": list(active_global_ids),
            "resolved_config_sha256": canonical_json_hash(resolved) if resolved else None,
            "view_manifest_hash": view_manifest_hash,
        },
    )
    result.update({
        "method": checkpoint_role,
        "checkpoint_role": checkpoint_role,
        "checkpoint": Path(checkpoint).name,
        "checkpoint_sha256": audit["sha256"],
        "checkpoint_audit": audit,
        "view_sha256": __import__("hashlib").sha256(Path(view).read_bytes()).hexdigest(),
        "resolved_config_sha256": canonical_json_hash(resolved) if resolved else None,
        "prediction_binding": {
            "checkpoint_sha256": audit["sha256"],
            "view_manifest_hash": result.get("manifest_hash"),
            "prediction_sha256": result.get("prediction_sha256"),
        },
    })
    write_json(output_summary, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ovtr-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--text-embedding", required=True)
    parser.add_argument("--image-embedding")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--active-global-ids", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-frames-per-video", type=int)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--motion-mode", default="none")
    parser.add_argument("--label-mode", default="complete")
    parser.add_argument("--alignment", action="store_true")
    parser.add_argument("--foundation-partial", action="store_true")
    parser.add_argument("--asset-check", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, _, _, _, _ = make_model(
        args.ovtr_root,
        args.config,
        args.text_embedding,
        args.image_embedding,
        _ids(args.active_global_ids),
        args.device,
        clip_len=2,
        motion_mode=args.motion_mode,
        label_mode=args.label_mode,
        alignment=args.alignment or bool(args.image_embedding),
    )
    audit = load_checkpoint(
        model,
        args.checkpoint,
        strict=not args.foundation_partial,
        allow_foundation_partial=args.foundation_partial,
    )
    summary = run_video_inference(
        model,
        args.view,
        args.image_root,
        args.output_jsonl,
        args.split,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames_per_video,
        input_size=(args.input_width, args.input_height),
    )
    summary.update({
        "method": "asset_check" if args.asset_check else "inference",
        "checkpoint_audit": audit,
        "checkpoint_role": "asset_check_only" if args.asset_check else "trained_or_runtime",
    })
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
