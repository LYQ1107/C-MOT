"""Audit an existing prediction artifact without retraining or changing it."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from cmot.manifest import sha256_file, write_json


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix0, iy0 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    ix1, iy1 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_l = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_r = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return inter / max(area_l + area_r - inter, 1e-8)


def _read_prediction_records(path: str) -> Tuple[Dict[str, dict], dict]:
    records = {}
    metadata = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("record_type") == "metadata":
                metadata.update(value.get("metadata", value))
                continue
            frame_key = value.get("frame_key")
            if not frame_key:
                raise ValueError("prediction line %d has no frame_key" % line_number)
            if frame_key in records:
                raise ValueError("duplicate frame_key %s" % frame_key)
            seen = set()
            for prediction in value.get("predictions", []):
                track_id = int(prediction["track_id"])
                if track_id in seen:
                    raise ValueError("duplicate track_id %s on %s" % (track_id, frame_key))
                seen.add(track_id)
                box = prediction.get("bbox_xyxy")
                if box is None or len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                    raise ValueError("invalid box on %s" % frame_key)
                if float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
                    raise ValueError("non-positive box on %s" % frame_key)
            records[str(frame_key)] = value
    return records, metadata


def diagnose_predictions(
    predictions: str,
    view: str,
    output: str,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    duplicate_iou: float = 0.85,
) -> dict:
    records, metadata = _read_prediction_records(predictions)
    view_payload = json.loads(Path(view).read_text(encoding="utf-8"))
    videos = sorted(view_payload.get("videos", []), key=lambda value: str(value["video_id"]))
    if max_videos is not None:
        videos = videos[: int(max_videos)]
    selected_video_ids = {str(v["video_id"]) for v in videos}
    records = {
        key: value for key, value in records.items()
        if str(value.get("video_id", "")) in selected_video_ids
    }
    expected = set()
    by_class = Counter()
    by_video = Counter()
    frame_prediction_counts = []
    duplicate_frames = 0
    duplicate_pairs = 0
    observed_tracks = defaultdict(list)
    gt_by_class = Counter()
    for video in videos:
        frames = sorted(video.get("frames", []), key=lambda value: (int(value["frame_index"]), value["frame_key"]))
        if max_frames_per_video is not None:
            frames = frames[: int(max_frames_per_video)]
        for frame in frames:
            frame_key = str(frame["frame_key"])
            expected.add(frame_key)
            for ann in frame.get("annotations", []):
                if not int(ann.get("iscrowd", 0)) and ann.get("label_status") != "ignore":
                    gt_by_class[str(int(ann.get("global_semantic_id", -1)))] += 1
            predictions_for_frame = list(records.get(frame_key, {}).get("predictions", []))
            frame_prediction_counts.append(len(predictions_for_frame))
            by_video[str(video["video_id"])] += len(predictions_for_frame)
            local_pairs = 0
            for index, prediction in enumerate(predictions_for_frame):
                gid = str(int(prediction.get("global_id", -1)))
                by_class[gid] += 1
                observed_tracks[(str(video["video_id"]), int(prediction["track_id"]))].append(int(frame["frame_index"]))
                for other in predictions_for_frame[:index]:
                    if int(other.get("global_id", -1)) == int(prediction.get("global_id", -1)) and _iou(other["bbox_xyxy"], prediction["bbox_xyxy"]) >= float(duplicate_iou):
                        local_pairs += 1
            if local_pairs:
                duplicate_frames += 1
                duplicate_pairs += local_pairs
    missing = sorted(expected - set(records))
    extra = sorted(set(records) - expected)
    gaps = []
    for values in observed_tracks.values():
        values = sorted(set(values))
        gaps.extend(max(0, right - left - 1) for left, right in zip(values, values[1:]))
    result = {
        "schema_version": "cmot.asset-diagnosis.v1",
        "method": "asset_check",
        "execution_status": "COMPLETE",
        "evaluation_scope": "asset_diagnosis",
        "view": {"basename": Path(view).name, "sha256": sha256_file(view), "manifest_hash": view_payload.get("manifest_hash")},
        "predictions": {"basename": Path(predictions).name, "sha256": sha256_file(predictions)},
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "checkpoint_role": metadata.get("checkpoint_role", "asset_check_only"),
        "active_global_ids": metadata.get("active_global_ids", view_payload.get("active_global_ids", [])),
        "videos": len(videos),
        "frames": len(expected),
        "missing_prediction_frames": len(missing),
        "extra_prediction_frames": len(extra),
        "predictions": int(sum(frame_prediction_counts)),
        "prediction_count_by_global_id": dict(sorted(by_class.items())),
        "gt_count_by_global_id": dict(sorted(gt_by_class.items())),
        "independent_video_count": len({str(v["video_id"]) for v in videos}),
        "mean_predictions_per_frame": None if not frame_prediction_counts else sum(frame_prediction_counts) / float(len(frame_prediction_counts)),
        "duplicate_frame_count": duplicate_frames,
        "duplicate_pair_count": duplicate_pairs,
        "track_count": len(observed_tracks),
        "track_gap_count": sum(1 for gap in gaps if gap > 0),
        "max_track_gap_frames": max(gaps) if gaps else 0,
        "status": "OK" if not missing and not extra else "FRAME_SET_MISMATCH",
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-frames-per-video", type=int)
    parser.add_argument("--duplicate-iou", type=float, default=0.85)
    args = parser.parse_args()
    print(json.dumps(diagnose_predictions(
        args.predictions, args.view, args.output, args.max_videos,
        args.max_frames_per_video, args.duplicate_iou,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
