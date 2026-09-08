"""Runtime bridge from C-MOT protocols to the imported OVTR implementation."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    resolved_config: Optional[Mapping[str, Any]] = None,
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
    resolved = dict(resolved_config or {})
    # Resolve once, then assign the final values to the actual OVTR args and
    # model attributes.  Earlier V2 code set the parser defaults first and
    # accidentally let those values win over the YAML contract.
    motion_cfg = dict(resolved.get("motion", {}))
    inference_cfg = dict(resolved.get("inference", {}))
    supervision_cfg = dict(resolved.get("supervision", {}))
    distillation_cfg = dict(resolved.get("distillation", {}))
    if resolved:
        motion_mode = str(motion_cfg.get("mode", motion_mode))
        label_mode = "partial" if str(resolved.get("stage", "")).startswith("S1") or str(resolved.get("stage", "")).startswith("S2") else label_mode
        score_threshold = float(inference_cfg.get("birth_threshold", inference_cfg.get("score_threshold", score_threshold)))
        filter_threshold = float(inference_cfg.get("keep_threshold", inference_cfg.get("filter_threshold", filter_threshold)))
        miss_tolerance = int(inference_cfg.get("miss_tolerance", miss_tolerance))
        maximum_quantity = int(inference_cfg.get("maximum_quantity", maximum_quantity))
    args.score_thresh = [float(score_threshold)]
    args.filter_score_thresh = [float(filter_threshold)]
    args.miss_tolerance = [int(miss_tolerance)]
    cfg.cmot_protocol_role = str(resolved.get("protocol_role", "cil"))
    cfg.cmot_pl_iou_min = float(supervision_cfg.get("quality_iou_min", 0.5))
    cfg.cmot_lambda_gt = float(supervision_cfg.get("lambda_gt", 1.0))
    cfg.cmot_lambda_pl = float(supervision_cfg.get("lambda_pl", 0.25))
    cfg.cmot_pl_warmup_steps = int(supervision_cfg.get("pl_warmup_steps", 100))
    cfg.cmot_distillation_cfg = distillation_cfg
    cfg.cmot_motion_mode = motion_mode
    cfg.cmot_motion_loss_coef = float(motion_cfg.get("lambda_motion", 0.1))
    cfg.cmot_motion_velocity_limit = float(motion_cfg.get("velocity_limit", 1.25))
    cfg.cmot_motion_warmup_steps = int(motion_cfg.get("warmup_steps", 20))
    cfg.cmot_motion_detach_features = bool(motion_cfg.get("detach_features", True))
    cfg.cmot_motion_detach_reference = bool(motion_cfg.get("detach_reference", True))
    cfg.cmot_motion_max_dt = float(motion_cfg.get("max_dt", 2.0))
    cfg.cmot_inference_dedup_enabled = bool(inference_cfg.get("inference_dedup_enabled", True))
    cfg.cmot_birth_threshold = float(inference_cfg.get("birth_threshold", score_threshold))
    cfg.cmot_keep_threshold = float(inference_cfg.get("keep_threshold", filter_threshold))
    cfg.cmot_export_threshold = float(inference_cfg.get("export_threshold", score_threshold))
    cfg.cmot_duplicate_iou = float(inference_cfg.get("duplicate_iou", 0.85))
    cfg.cmot_duplicate_feature_cos = float(inference_cfg.get("duplicate_feature_cos", 0.95))
    cfg.cmot_dedup_new_new = bool(inference_cfg.get("dedup_new_new", True))
    cfg.cmot_dedup_new_track = bool(inference_cfg.get("dedup_new_track", True))
    cfg.cmot_merge_existing_ids = bool(inference_cfg.get("merge_existing_ids", False))
    cfg.cmot_label_mode = label_mode
    cfg.cmot_continual_cfg = {
        "motion_mode": motion_mode,
        "maximum_quantity": int(maximum_quantity),
        "birth_threshold": cfg.cmot_birth_threshold,
        "keep_threshold": cfg.cmot_keep_threshold,
        "export_threshold": cfg.cmot_export_threshold,
        "ious_thresh": cfg.cmot_duplicate_iou,
        "velocity_limit": cfg.cmot_motion_velocity_limit,
        "warmup_steps": cfg.cmot_motion_warmup_steps,
        "detach_features": cfg.cmot_motion_detach_features,
        "detach_reference": cfg.cmot_motion_detach_reference,
        "max_dt": cfg.cmot_motion_max_dt,
        "duplicate_iou": cfg.cmot_duplicate_iou,
        "duplicate_feature_cos": cfg.cmot_duplicate_feature_cos,
        "dedup_new_new": cfg.cmot_dedup_new_new,
        "dedup_new_track": cfg.cmot_dedup_new_track,
        "merge_existing_ids": cfg.cmot_merge_existing_ids,
    }
    cfg.cmot_runtime_config = resolved
    model, criterion = build_model(args, cfg)
    model.to(torch.device(device))
    if getattr(model, "track_base", None) is not None:
        actual = {
            "birth_threshold": float(model.track_base.birth_threshold),
            "keep_threshold": float(model.track_base.keep_threshold),
            "export_threshold": float(model.track_base.export_threshold),
            "miss_tolerance": int(model.track_base.miss_tolerance),
            "maximum_quantity": int(model.track_base.maximum_quantity),
        }
        expected = {
            "birth_threshold": cfg.cmot_birth_threshold,
            "keep_threshold": cfg.cmot_keep_threshold,
            "export_threshold": cfg.cmot_export_threshold,
            "miss_tolerance": miss_tolerance,
            "maximum_quantity": maximum_quantity,
        }
        for key, value in expected.items():
            if isinstance(value, float):
                if abs(actual[key] - value) > 1e-8:
                    raise AssertionError("runtime inference mapping failed for %s: %s != %s" % (key, actual[key], value))
            elif actual[key] != value:
                raise AssertionError("runtime inference mapping failed for %s: %s != %s" % (key, actual[key], value))
    return model, criterion, registry, args, cfg


def _state_dict(payload):
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dictionary")
    return payload.get("model", payload)


def load_checkpoint(
    model,
    path: str,
    strict: bool = True,
    allow_foundation_partial: bool = False,
    *,
    init_mode: Optional[str] = None,
    expected_metadata: Optional[Mapping[str, Any]] = None,
    allowed_missing_prefixes: Sequence[str] = (),
) -> dict:
    """Load a checkpoint with an explicit foundation/transfer/resume audit.

    strict and allow_foundation_partial remain accepted for existing
    asset-check callers. New repair_v2 callers use init_mode so a missing
    motion head cannot be silently accepted during stage transfer.
    """
    path_obj = Path(path)
    payload = torch.load(str(path_obj), map_location="cpu")
    if init_mode is None:
        init_mode = "resume" if strict else ("foundation" if allow_foundation_partial else "resume")
    init_mode = str(init_mode)
    if init_mode not in ("foundation", "stage_transfer", "resume"):
        raise ValueError("unknown checkpoint init_mode %s" % init_mode)
    state = _state_dict(payload)
    if not isinstance(state, dict):
        raise ValueError("checkpoint model state is not a dictionary")
    model_state = model.state_dict()
    missing = [key for key in model_state if key not in state]
    unexpected = [key for key in state if key not in model_state]
    mismatch = [key for key in model_state if key in state and tuple(model_state[key].shape) != tuple(state[key].shape)]
    if mismatch:
        raise ValueError("checkpoint shape mismatch: %s" % mismatch[:5])
    if init_mode == "resume" and (missing or unexpected):
        raise ValueError("resume checkpoint audit failed: missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    if init_mode == "stage_transfer":
        illegal_missing = [
            key for key in missing
            if not any(str(key).startswith(str(prefix)) for prefix in allowed_missing_prefixes)
        ]
        if illegal_missing or unexpected:
            raise ValueError(
                "stage-transfer checkpoint audit failed: missing=%s unexpected=%s"
                % (illegal_missing[:5], unexpected[:5])
            )
    if init_mode == "foundation" and not (allow_foundation_partial or not strict):
        raise ValueError("foundation checkpoint requires explicit partial-load flag")
    checkpoint_metadata = payload.get("cmot_metadata", {}) if isinstance(payload, dict) else {}
    metadata_mismatches = {}
    for key, expected in (expected_metadata or {}).items():
        actual = checkpoint_metadata.get(key, payload.get(key) if isinstance(payload, dict) else None)
        if actual != expected:
            metadata_mismatches[str(key)] = {"expected": expected, "actual": actual}
    if metadata_mismatches:
        raise ValueError("checkpoint metadata mismatch: %s" % metadata_mismatches)
    compatible = {key: value for key, value in state.items() if key in model_state}
    model.load_state_dict(compatible, strict=False)
    return {
        "path_basename": path_obj.name,
        "sha256": sha256_file(str(path_obj)),
        "bytes": path_obj.stat().st_size,
        "strict": bool(init_mode == "resume"),
        "init_mode": init_mode,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": mismatch,
        "foundation_partial": bool(init_mode == "foundation"),
        "allowed_missing_prefixes": list(allowed_missing_prefixes),
        "metadata_mismatches": metadata_mismatches,
        "checkpoint_metadata": checkpoint_metadata,
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
    metadata: Optional[Mapping[str, Any]] = None,
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
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()
    with partial.open("w", encoding="utf-8") as handle:
        if metadata:
            handle.write(json.dumps({"record_type": "metadata", "metadata": dict(metadata)}, sort_keys=True) + "\n")
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
                record["source_video_uid"] = str(video.get("source_video_uid") or video.get("source_video_id") or video["video_id"])
                record["video_uid"] = record["source_video_uid"]
                record["frame_index"] = int(frame["frame_index"])
                record["frame_uid"] = str(frame.get("frame_uid") or frame["frame_key"])
                record["width"] = int(frame["width"])
                record["height"] = int(frame["height"])
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                frame_count += 1
                prediction_count += len(record.get("predictions", []))
    os.replace(str(partial), str(destination))
    runtime_stats = model.consume_runtime_stats() if hasattr(model, "consume_runtime_stats") else {}
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
        "prediction_sha256": sha256_file(str(destination)),
        "runtime_stats": runtime_stats,
    }
