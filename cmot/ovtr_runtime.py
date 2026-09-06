"""Runtime bridge from C-MOT protocols to the imported OVTR implementation."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .class_registry import tao_bdd_registry
from .manifest import sha256_file, write_json


def _ensure_ovtr_imports(ovtr_root: str) -> None:
    root = str(Path(ovtr_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def make_model(
    ovtr_root: str,
    config_file: str,
    text_embedding: str,
    image_embedding: Optional[str],
    active_global_ids: Sequence[int],
    device: str,
    clip_len: int = 2,
    motion_mode: str = "none",
    label_mode: str = "complete",
    score_threshold: float = 0.19,
    filter_threshold: float = 0.19,
    miss_tolerance: int = 5,
    maximum_quantity: int = 160,
    alignment: bool = True,
):
    _ensure_ovtr_imports(ovtr_root)
    from main import get_args_parser
    from models import build_model
    from util.slconfig import SLConfig

    args = get_args_parser().parse_args([])
    args.device = device
    args.sampler_lengths = [int(clip_len)]
    args.batch_size = 1
    args.two_stage = True
    args.with_box_refine = True
    args.calculate_negative_samples = True
    args.max_len = 250
    args.score_thresh = [float(score_threshold)]
    args.filter_score_thresh = [float(filter_threshold)]
    args.miss_tolerance = [int(miss_tolerance)]
    cfg = SLConfig.fromfile(config_file)
    cfg.train_with_artificial_img_seqs = False
    cfg.use_checkpoint_track = False
    cfg.Clip_text_embeddings = str(Path(text_embedding).resolve())
    cfg.Clip_image_embeddings = str(Path(image_embedding).resolve()) if alignment and image_embedding else None
    registry = tao_bdd_registry()
    names = []
    for global_id in active_global_ids:
        names.append(registry.class_name_for_global(int(global_id)))
    registry.set_active(names)
    cfg.cmot_class_registry = registry
    cfg.cmot_motion_mode = motion_mode
    cfg.cmot_label_mode = label_mode
    cfg.cmot_continual_cfg = {
        "motion_mode": motion_mode,
        "maximum_quantity": int(maximum_quantity),
        "ious_thresh": 0.45,
    }
    model, criterion = build_model(args, cfg)
    model.to(torch.device(device))
    return model, criterion, registry, args, cfg


def _state_dict(payload):
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dictionary")
    return payload.get("model", payload)


def load_checkpoint(model, path: str, strict: bool = True, allow_foundation_partial: bool = False) -> dict:
    path_obj = Path(path)
    state = _state_dict(torch.load(str(path_obj), map_location="cpu"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint model state is not a dictionary")
    model_state = model.state_dict()
    missing = [key for key in model_state if key not in state]
    unexpected = [key for key in state if key not in model_state]
    mismatch = [key for key in model_state if key in state and tuple(model_state[key].shape) != tuple(state[key].shape)]
    if mismatch:
        raise ValueError("checkpoint shape mismatch: %s" % mismatch[:5])
    if strict and (missing or unexpected):
        raise ValueError("strict checkpoint audit failed: missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    if not strict and not allow_foundation_partial and (missing or unexpected):
        raise ValueError("partial checkpoint requires explicit foundation flag")
    compatible = {key: value for key, value in state.items() if key in model_state}
    model.load_state_dict(compatible, strict=False)
    return {
        "path_basename": path_obj.name,
        "sha256": sha256_file(str(path_obj)),
        "bytes": path_obj.stat().st_size,
        "strict": bool(strict),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": mismatch,
        "foundation_partial": bool(allow_foundation_partial),
    }


def iter_view_videos(view_path: str, split: str, max_videos: Optional[int] = None, video_ids: Optional[Sequence[str]] = None):
    with open(view_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    wanted = set(video_ids or [])
    videos = [v for v in payload.get("videos", []) if v.get("split") == split]
    videos = sorted(videos, key=lambda value: value["video_id"])
    if wanted:
        videos = [v for v in videos if v["video_id"] in wanted]
    if max_videos is not None:
        videos = videos[: int(max_videos)]
    return payload, videos


@torch.no_grad()
def run_video_inference(
    model,
    view_path: str,
    image_root: str,
    output_jsonl: str,
    split: str,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    input_size: Tuple[int, int] = (640, 360),
    video_ids: Optional[Sequence[str]] = None,
) -> dict:
    # Import after make_model() has installed OVTR's bundled detectron2
    # compatibility package on sys.path.  Importing the dataset at module
    # load time lets a site-wide detectron2 win and hides
    # matched_boxlist_iou, which OVTR's source expects.
    _ensure_ovtr_imports(str(Path(__file__).resolve().parents[1] / "ovtr"))
    from .data.real_video_dataset import _read_image

    payload, videos = iter_view_videos(view_path, split, max_videos, video_ids)
    destination = Path(output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    frame_count = 0
    prediction_count = 0
    selected_videos = []
    with destination.open("w", encoding="utf-8") as handle:
        for video in videos:
            selected_videos.append(video["video_id"])
            runtime_state = None
            frames = sorted(video.get("frames", []), key=lambda value: (int(value["frame_index"]), value["frame_key"]))
            if max_frames_per_video is not None:
                frames = frames[: int(max_frames_per_video)]
            for local_frame_id, frame in enumerate(frames):
                image = _read_image(Path(image_root) / (video.get("image_root") or "") / frame["file_name"], input_size)
                runtime_state, record, _ = model.inference_video_frame(
                    {"imgs": [image]},
                    runtime_state,
                    {
                        "frame_id": local_frame_id,
                        "frame_key": frame["frame_key"],
                        "target_size": (int(frame["height"]), int(frame["width"])),
                        "timestamp_s": frame.get("timestamp_s"),
                    },
                )
                record["video_id"] = video["video_id"]
                record["frame_index"] = int(frame["frame_index"])
                record["width"] = int(frame["width"])
                record["height"] = int(frame["height"])
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                frame_count += 1
                prediction_count += len(record.get("predictions", []))
    return {
        "view": Path(view_path).name,
        "split": split,
        "videos": len(selected_videos),
        "video_ids": selected_videos,
        "frames": frame_count,
        "predictions": prediction_count,
        "input_size": list(input_size),
        "output_jsonl": destination.name,
        "manifest_hash": payload.get("manifest_hash"),
    }
