"""Conservative teacher-track filtering and legal-GT calibration."""

import hashlib
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix0, iy0 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    ix1, iy1 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_l = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_r = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return inter / max(area_l + area_r - inter, 1e-8)


def wilson_lower_bound(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if int(total) <= 0:
        return 0.0
    n = float(total)
    phat = float(successes) / n
    denominator = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    spread = z * math.sqrt(max(0.0, phat * (1.0 - phat) / n + z * z / (4.0 * n * n)))
    return (centre - spread) / denominator


def _record_key(record: Mapping[str, object], prediction: Mapping[str, object]) -> Tuple[str, int, int]:
    return (
        str(record.get("video_uid") or record.get("video_id") or ""),
        int(prediction.get("global_id", -1)),
        int(prediction.get("track_id", -1)),
    )


def calibrate_threshold(
    prediction_records: Iterable[Mapping[str, object]],
    gt_by_frame: Mapping[str, Sequence[Mapping[str, object]]],
    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    iou_min: float = 0.5,
    min_predictions: int = 30,
    min_videos: int = 3,
    wilson_lower_min: float = 0.8,
    old_global_ids: Sequence[int] = (),
) -> dict:
    """Select the lowest supported threshold meeting the Wilson guard."""
    old = {int(v) for v in old_global_ids}
    records = list(prediction_records)
    candidates = []
    for threshold in sorted(float(v) for v in thresholds):
        total = correct = 0
        videos = set()
        by_frame = defaultdict(list)
        for record in records:
            frame_key = str(record.get("frame_key", ""))
            for pred in record.get("predictions", []):
                if int(pred.get("global_id", -1)) not in old or float(pred.get("score", -1)) < threshold:
                    continue
                total += 1
                videos.add(str(record.get("video_uid") or record.get("video_id") or frame_key.rsplit("/", 1)[0]))
                by_frame[frame_key].append(pred)
        for frame_key, predictions in by_frame.items():
            used = set()
            gt = [a for a in gt_by_frame.get(frame_key, []) if int(a.get("global_semantic_id", -1)) in old]
            for prediction in sorted(predictions, key=lambda value: -float(value.get("score", 0.0))):
                matches = sorted(
                    ((index, _iou(prediction["bbox_xyxy"], ann["bbox_xyxy"]))
                     for index, ann in enumerate(gt)
                     if index not in used and int(ann.get("global_semantic_id", -1)) == int(prediction.get("global_id", -1))),
                    key=lambda value: value[1], reverse=True,
                )
                if matches and matches[0][1] >= float(iou_min):
                    used.add(matches[0][0])
                    correct += 1
        lower = wilson_lower_bound(correct, total)
        candidates.append({
            "threshold": threshold,
            "predictions": total,
            "correct": correct,
            "videos": len(videos),
            "precision": None if total == 0 else correct / float(total),
            "wilson_lower": lower,
            "eligible": bool(total >= int(min_predictions) and len(videos) >= int(min_videos) and lower >= float(wilson_lower_min)),
        })
    selected = next((value for value in candidates if value["eligible"]), None)
    if selected is None:
        return {
            "status": "PL_DISABLED_LOW_SUPPORT_OR_QUALITY",
            "selected": None,
            "candidates": candidates,
            "reason": "no threshold met minimum prediction/video support and Wilson lower bound",
        }
    return {"status": "OK", "selected": selected, "candidates": candidates}


def filter_prediction_records(
    prediction_records: Iterable[Mapping[str, object]],
    teacher_checkpoint_sha256: Optional[str],
    old_global_ids: Sequence[int],
    thresholds: Mapping[int, float],
    per_class_caps: Mapping[int, int],
    total_cap: int = 40,
    min_segment_frames: int = 3,
    max_gap_s: float = 1.0,
    conflict_gt_by_frame: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
    conflict_iou: float = 0.7,
    duplicate_iou: float = 0.7,
) -> Tuple[List[dict], dict]:
    """Filter predictions into accepted track segments with q25 reliability."""
    if not teacher_checkpoint_sha256 or str(teacher_checkpoint_sha256) in ("unknown", "None"):
        return [], {"status": "PL_DISABLED_MISSING_TEACHER_HASH", "accepted": 0}
    old = {int(v) for v in old_global_ids}
    records = list(prediction_records)
    by_track = defaultdict(list)
    for record in records:
        for prediction in record.get("predictions", []):
            gid = int(prediction.get("global_id", -1))
            if gid not in old or float(prediction.get("score", -1)) < float(thresholds.get(gid, 1.0)):
                continue
            value = dict(prediction)
            value["frame_key"] = str(record.get("frame_key", ""))
            value["video_uid"] = str(record.get("video_uid") or record.get("video_id") or "")
            value["frame_index"] = int(record.get("frame_index", 0))
            value["timestamp_s"] = record.get("timestamp_s")
            by_track[_record_key(record, prediction)].append(value)
    accepted = []
    rejected = Counter()
    for key, track in by_track.items():
        track.sort(key=lambda value: (value["frame_index"], value["frame_key"]))
        segments = []
        current = []
        for value in track:
            if current:
                previous = current[-1]
                frame_gap = int(value["frame_index"]) - int(previous["frame_index"])
                times = value.get("timestamp_s") is not None and previous.get("timestamp_s") is not None
                time_gap = float(value["timestamp_s"]) - float(previous["timestamp_s"]) if times else 0.0
                if frame_gap > 1 or (times and (time_gap > float(max_gap_s) or time_gap <= 0)):
                    segments.append(current)
                    current = []
            current.append(value)
        if current:
            segments.append(current)
        for segment in segments:
            if len(segment) < int(min_segment_frames):
                rejected["short_segment"] += len(segment)
                continue
            scores = sorted(float(v["score"]) for v in segment)
            reliability = scores[max(0, int(math.ceil(0.25 * len(scores))) - 1)]
            segment_id = hashlib.sha256(("%s|%s|%s|%s" % (key[0], key[1], key[2], segment[0]["frame_index"])).encode("utf-8")).hexdigest()[:20]
            for value in segment:
                value["pl_segment_id"] = segment_id
                value["reliability"] = reliability
                if conflict_gt_by_frame:
                    conflict = any(
                        int(gt.get("global_semantic_id", -1)) != int(value["global_id"])
                        and _iou(gt["bbox_xyxy"], value["bbox_xyxy"]) >= float(conflict_iou)
                        for gt in conflict_gt_by_frame.get(value["frame_key"], [])
                    )
                    if conflict:
                        rejected["gt_conflict"] += 1
                        continue
                accepted.append(value)
    by_frame = defaultdict(list)
    for value in accepted:
        by_frame[value["frame_key"]].append(value)
    deduped = []
    for frame_key, values in by_frame.items():
        kept = []
        for value in sorted(values, key=lambda x: (-float(x["reliability"]), -float(x["score"]), str(x["pl_segment_id"]))):
            if any(int(other["global_id"]) == int(value["global_id"]) and _iou(other["bbox_xyxy"], value["bbox_xyxy"]) >= float(duplicate_iou) for other in kept):
                rejected["same_class_duplicate"] += 1
            else:
                kept.append(value)
        deduped.extend(kept)
    deduped.sort(key=lambda value: (-float(value["reliability"]), -float(value["score"]), value["frame_key"], int(value["track_id"])))
    selected = []
    per_class = Counter()
    for value in deduped:
        gid = int(value["global_id"])
        if per_class[gid] >= int(per_class_caps.get(gid, total_cap)):
            rejected["per_class_cap"] += 1
            continue
        if len(selected) >= int(total_cap):
            rejected["total_cap"] += 1
            continue
        per_class[gid] += 1
        selected.append(value)
    output = []
    for value in selected:
        output.append({
            "track_id": int(value["track_id"]),
            "global_id": int(value["global_id"]),
            "score": float(value["score"]),
            "bbox_xyxy": [float(v) for v in value["bbox_xyxy"]],
            "frame_key": value["frame_key"],
            "video_uid": value["video_uid"],
            "frame_index": int(value["frame_index"]),
            "timestamp_s": value.get("timestamp_s"),
            "pl_segment_id": value["pl_segment_id"],
            "reliability": float(value["reliability"]),
            "teacher_checkpoint_sha256": str(teacher_checkpoint_sha256),
        })
    return output, {
        "status": "OK",
        "accepted": len(output),
        "accepted_by_class": dict(sorted(per_class.items())),
        "rejected": dict(sorted(rejected.items())),
        "teacher_checkpoint_sha256": str(teacher_checkpoint_sha256),
    }
