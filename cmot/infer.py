"""CLI for asset checks and raw real-video prediction generation."""

import argparse
import json
from pathlib import Path

from .ovtr_runtime import load_checkpoint, make_model, run_video_inference
from .manifest import write_json


def _ids(value: str):
    return [int(item) for item in value.split(",") if item]


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
    args = parser.parse_args()
    model, _, _, _, _ = make_model(
        args.ovtr_root,
        args.config,
        args.text_embedding,
        args.image_embedding,
        _ids(args.active_global_ids),
        "cuda",
        clip_len=2,
        motion_mode=args.motion_mode,
        label_mode=args.label_mode,
        alignment=args.alignment or bool(args.image_embedding),
    )
    audit = load_checkpoint(model, args.checkpoint, strict=not args.foundation_partial, allow_foundation_partial=args.foundation_partial)
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

