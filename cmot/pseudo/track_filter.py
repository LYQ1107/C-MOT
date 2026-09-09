"""Conservative teacher-track calibration and segment-level QPL filtering."""

import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
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
        int(prediction.get("teacher_track_id", prediction.get("track_id", -1))),
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
                    (
                        (index, _iou(prediction["bbox_xyxy"], ann["bbox_xyxy"]))
                        for index, ann in enumerate(gt)
                        if index not in used
                        and int(ann.get("global_semantic_id", -1)) == int(prediction.get("global_id", -1))
                    ),
                    key=lambda value: value[1],
                    reverse=True,
                )
                if matches and matches[0][1] >= float(iou_min):
                    used.add(matches[0][0])
                    correct += 1
        lower = wilson_lower_bound(correct, total)
        candidates.append(
            {
                "threshold": threshold,
                "predictions": total,
                "correct": correct,
                "videos": len(videos),
                "precision": None if total == 0 else correct / float(total),
                "wilson_lower": lower,
                "eligible": bool(
                    total >= int(min_predictions)
                    and len(videos) >= int(min_videos)
                    and lower >= float(wilson_lower_min)
                ),
            }
        )
    selected = next((value for value in candidates if value["eligible"]), None)
    if selected is None:
        return {
            "status": "PL_DISABLED_LOW_SUPPORT_OR_QUALITY",
            "selected": None,
            "candidates": candidates,
            "reason": "no threshold met minimum prediction/video support and Wilson lower bound",
        }
    return {"status": "OK", "selected": selected, "candidates": candidates}


@dataclass
class PseudoTrackSegment:
    segment_id: str
    video_uid: str
    global_id: int
    teacher_track_id: int
    frames: List[dict]
    reliability: float
    mean_score: float

    @property
    def length(self) -> int:
        return len(self.frames)

    def as_audit_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "video_uid": self.video_uid,
            "global_id": int(self.global_id),
            "teacher_track_id": int(self.teacher_track_id),
            "length": int(self.length),
            "reliability": float(self.reliability),
            "mean_score": float(self.mean_score),
            "frame_keys": [str(frame.get("frame_key", "")) for frame in self.frames],
        }


def _segment_id(video_uid: str, global_id: int, teacher_track_id: int, frames: Sequence[Mapping[str, object]]) -> str:
    first = str(frames[0].get("frame_key", frames[0].get("frame_index", 0)))
    identity = "%s|%d|%d|%s" % (video_uid, int(global_id), int(teacher_track_id), first)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _make_segment(track: Sequence[dict]) -> PseudoTrackSegment:
    frames = [dict(value) for value in track]
    frames.sort(key=lambda value: (int(value.get("frame_index", 0)), str(value.get("frame_key", ""))))
    scores = sorted(float(value["score"]) for value in frames)
    q25_index = max(0, min(len(scores) - 1, int(math.ceil(0.25 * len(scores))) - 1))
    video_uid = str(frames[0]["video_uid"])
    global_id = int(frames[0]["global_id"])
    teacher_track_id = int(frames[0]["teacher_track_id"])
    return PseudoTrackSegment(
        segment_id=_segment_id(video_uid, global_id, teacher_track_id, frames),
        video_uid=video_uid,
        global_id=global_id,
        teacher_track_id=teacher_track_id,
        frames=frames,
        reliability=float(scores[q25_index]),
        mean_score=float(sum(scores) / len(scores)),
    )


def _contiguous_chunks(track: Sequence[dict], max_gap_s: float) -> List[List[dict]]:
    ordered = sorted(
        (dict(value) for value in track),
        key=lambda value: (int(value.get("frame_index", 0)), str(value.get("frame_key", ""))),
    )
    chunks: List[List[dict]] = []
    current: List[dict] = []
    for value in ordered:
        contiguous = False
        if current:
            previous = current[-1]
            same_identity = (
                str(value.get("video_uid", "")) == str(previous.get("video_uid", ""))
                and int(value.get("global_id", -1)) == int(previous.get("global_id", -1))
                and int(value.get("teacher_track_id", -1)) == int(previous.get("teacher_track_id", -1))
            )
            frame_gap = int(value.get("frame_index", 0)) - int(previous.get("frame_index", 0))
            timestamps_available = value.get("timestamp_s") is not None and previous.get("timestamp_s") is not None
            time_valid = True
            if timestamps_available:
                try:
                    dt = float(value["timestamp_s"]) - float(previous["timestamp_s"])
                    time_valid = math.isfinite(dt) and 0.0 < dt <= float(max_gap_s)
                except (TypeError, ValueError):
                    time_valid = False
            contiguous = same_identity and frame_gap == 1 and time_valid
        if current and not contiguous:
            chunks.append(current)
            current = []
        current.append(value)
    if current:
        chunks.append(current)
    return chunks


def _split_contiguous_track(
    track: Sequence[dict],
    min_segment_frames: int,
    max_gap_s: float,
) -> List[PseudoTrackSegment]:
    """Split a teacher track at every temporal or identity discontinuity."""
    return [
        _make_segment(chunk)
        for chunk in _contiguous_chunks(track, max_gap_s)
        if len(chunk) >= int(min_segment_frames)
    ]


def _segments_are_duplicates(
    left: PseudoTrackSegment,
    right: PseudoTrackSegment,
    duplicate_iou: float,
) -> bool:
    if left.video_uid != right.video_uid or int(left.global_id) != int(right.global_id):
        return False
    right_by_frame = {str(frame.get("frame_key", "")): frame for frame in right.frames}
    common = [
        (frame, right_by_frame[str(frame.get("frame_key", ""))])
        for frame in left.frames
        if str(frame.get("frame_key", "")) in right_by_frame
    ]
    if not common:
        return False
    return all(_iou(left_frame["bbox_xyxy"], right_frame["bbox_xyxy"]) >= float(duplicate_iou) for left_frame, right_frame in common)


def _limit_segment_length(segment: PseudoTrackSegment, max_segment_frames: Optional[int]) -> PseudoTrackSegment:
    if max_segment_frames is None or int(max_segment_frames) <= 0 or segment.length <= int(max_segment_frames):
        return segment
    limit = int(max_segment_frames)
    best = max(
        range(segment.length),
        key=lambda index: (-float(segment.frames[index]["score"]), int(segment.frames[index]["frame_index"]), str(segment.frames[index].get("frame_key", ""))),
    )
    start = min(max(0, best - limit // 2), segment.length - limit)
    limited = _make_segment(segment.frames[start : start + limit])
    return PseudoTrackSegment(
        segment_id=segment.segment_id,
        video_uid=segment.video_uid,
        global_id=segment.global_id,
        teacher_track_id=segment.teacher_track_id,
        frames=limited.frames,
        reliability=limited.reliability,
        mean_score=limited.mean_score,
    )


def _segment_priority(segment: PseudoTrackSegment) -> Tuple[float, float, int, str]:
    return (-float(segment.reliability), -float(segment.mean_score), -int(segment.length), str(segment.segment_id))


def _select_segments(
    candidates: Sequence[PseudoTrackSegment],
    total_segment_cap: int,
    per_class_segment_cap: Mapping[int, int],
    total_frame_cap: Optional[int],
) -> Tuple[List[PseudoTrackSegment], Counter]:
    """Select segments by class/source round-robin under both budgets."""
    grouped: Dict[int, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
    for segment in sorted(candidates, key=_segment_priority):
        grouped[int(segment.global_id)][str(segment.video_uid)].append(segment)
    class_order = sorted(grouped)
    source_order = {gid: sorted(grouped[gid]) for gid in class_order}
    selected: List[PseudoTrackSegment] = []
    selected_ids = set()
    selected_by_class = Counter()
    rejected = Counter()
    selected_frames = 0
    while True:
        progress = False
        if len(selected) >= int(total_segment_cap):
            rejected["rejected_segment_budget"] += sum(
                len(queue) for by_source in grouped.values() for queue in by_source.values()
            )
            break
        for global_id in class_order:
            if selected_by_class[global_id] >= int(per_class_segment_cap.get(global_id, total_segment_cap)):
                for queue in grouped[global_id].values():
                    rejected["rejected_segment_budget"] += len(queue)
                    queue.clear()
                continue
            for source_uid in source_order[global_id]:
                queue = grouped[global_id][source_uid]
                if not queue:
                    continue
                if selected_by_class[global_id] >= int(per_class_segment_cap.get(global_id, total_segment_cap)):
                    rejected["rejected_segment_budget"] += len(queue)
                    queue.clear()
                    continue
                segment = queue.popleft()
                selected_ids.add(segment.segment_id)
                progress = True
                if total_frame_cap is not None and selected_frames + segment.length > int(total_frame_cap):
                    rejected["rejected_frame_budget"] += 1
                    continue
                selected.append(segment)
                selected_by_class[global_id] += 1
                selected_frames += segment.length
                if len(selected) >= int(total_segment_cap):
                    break
            if len(selected) >= int(total_segment_cap):
                break
        if not progress:
            break
    # Queues emptied by a class/segment cap are already counted.  Any queue
    # left after a frame-budget-only pass is also an explicit rejection.
    for by_source in grouped.values():
        for queue in by_source.values():
            if queue:
                rejected["rejected_frame_budget"] += len(queue)
                queue.clear()
    return selected, rejected


def filter_prediction_records(
    prediction_records: Iterable[Mapping[str, object]],
    teacher_checkpoint_sha256: Optional[str],
    old_global_ids: Sequence[int],
    thresholds: Mapping[int, float],
    per_class_segment_caps: Mapping[int, int],
    total_segment_cap: int = 40,
    total_frame_cap: Optional[int] = 256,
    max_segment_frames: Optional[int] = 8,
    min_segment_frames: int = 3,
    max_gap_s: float = 1.0,
    conflict_gt_by_frame: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
    conflict_iou: float = 0.7,
    duplicate_iou: float = 0.7,
) -> Tuple[List[dict], dict]:
    """Filter predictions into deterministic, quality-ranked track segments."""
    empty_audit = {
        "status": "OK",
        "candidate_segments": 0,
        "selected_segments": 0,
        "selected_frame_predictions": 0,
        "selected_segments_by_class": {},
        "selected_frames_by_class": {},
        "selected_source_videos_by_class": {},
        "rejected_short_segments": 0,
        "rejected_gt_conflict_segments": 0,
        "rejected_duplicate_segments": 0,
        "rejected_segment_budget": 0,
        "rejected_frame_budget": 0,
    }
    if not teacher_checkpoint_sha256 or str(teacher_checkpoint_sha256) in ("unknown", "None"):
        empty_audit.update({"status": "PL_DISABLED_MISSING_TEACHER_HASH"})
        return [], empty_audit

    old = {int(v) for v in old_global_ids}
    records = list(prediction_records)
    by_track: Dict[Tuple[str, int, int], List[dict]] = defaultdict(list)
    for record in records:
        for prediction in record.get("predictions", []):
            global_id = int(prediction.get("global_id", -1))
            if global_id not in old or float(prediction.get("score", -1)) < float(thresholds.get(global_id, 1.0)):
                continue
            try:
                bbox = [float(v) for v in prediction["bbox_xyxy"]]
                if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            video_uid = str(record.get("video_uid") or record.get("video_id") or "")
            value = dict(prediction)
            value.update(
                {
                    "frame_key": str(record.get("frame_key", "")),
                    "video_uid": video_uid,
                    "frame_index": int(record.get("frame_index", 0)),
                    "timestamp_s": record.get("timestamp_s"),
                    "global_id": global_id,
                    "teacher_track_id": int(prediction.get("teacher_track_id", prediction.get("track_id", -1))),
                    "bbox_xyxy": bbox,
                    "score": float(prediction.get("score", 0.0)),
                }
            )
            by_track[_record_key(record, value)].append(value)

    rejected = Counter()
    candidates: List[PseudoTrackSegment] = []
    for _, track in sorted(by_track.items(), key=lambda item: repr(item[0])):
        for contiguous in _contiguous_chunks(track, max_gap_s):
            if len(contiguous) < int(min_segment_frames):
                rejected["rejected_short_segments"] += 1
                continue
            clean: List[dict] = []
            had_conflict = False
            for value in contiguous:
                conflict = False
                if conflict_gt_by_frame:
                    conflict = any(
                        int(gt.get("global_semantic_id", -1)) != int(value["global_id"])
                        and _iou(gt["bbox_xyxy"], value["bbox_xyxy"]) >= float(conflict_iou)
                        for gt in conflict_gt_by_frame.get(value["frame_key"], [])
                    )
                if conflict:
                    had_conflict = True
                    if clean:
                        if len(clean) >= int(min_segment_frames):
                            candidates.append(_limit_segment_length(_make_segment(clean), max_segment_frames))
                        else:
                            rejected["rejected_short_segments"] += 1
                        clean = []
                    continue
                clean.append(value)
            if had_conflict:
                rejected["rejected_gt_conflict_segments"] += 1
            if clean:
                if len(clean) >= int(min_segment_frames):
                    candidates.append(_limit_segment_length(_make_segment(clean), max_segment_frames))
                else:
                    rejected["rejected_short_segments"] += 1

    candidate_count = len(candidates)
    candidates.sort(key=_segment_priority)
    deduped: List[PseudoTrackSegment] = []
    for segment in candidates:
        if any(_segments_are_duplicates(segment, other, duplicate_iou) for other in deduped):
            rejected["rejected_duplicate_segments"] += 1
            continue
        deduped.append(segment)

    selected, budget_rejected = _select_segments(
        deduped,
        int(total_segment_cap),
        {int(key): int(value) for key, value in per_class_segment_caps.items()},
        None if total_frame_cap is None else int(total_frame_cap),
    )
    rejected.update(budget_rejected)

    selected_by_class = Counter(int(segment.global_id) for segment in selected)
    frames_by_class = Counter()
    videos_by_class: Dict[str, set] = defaultdict(set)
    output: List[dict] = []
    # Flattening is deliberately the final operation, after segment dedup and
    # both budgets have been applied.
    for segment in selected:
        for frame in sorted(segment.frames, key=lambda value: (int(value["frame_index"]), str(value["frame_key"]))):
            global_id = int(frame["global_id"])
            frames_by_class[global_id] += 1
            videos_by_class[str(global_id)].add(str(segment.video_uid))
            output.append(
                {
                    "track_id": int(frame.get("track_id", segment.teacher_track_id)),
                    "teacher_track_id": int(segment.teacher_track_id),
                    "global_id": global_id,
                    "score": float(frame["score"]),
                    "bbox_xyxy": [float(value) for value in frame["bbox_xyxy"]],
                    "frame_key": str(frame["frame_key"]),
                    "video_uid": str(segment.video_uid),
                    "frame_index": int(frame["frame_index"]),
                    "timestamp_s": frame.get("timestamp_s"),
                    "pl_segment_id": str(segment.segment_id),
                    "segment_length": int(segment.length),
                    "segment_reliability": float(segment.reliability),
                    "segment_mean_score": float(segment.mean_score),
                    # Keep the old field for downstream compatibility while
                    # making the segment-specific names the audit contract.
                    "reliability": float(segment.reliability),
                    "teacher_checkpoint_sha256": str(teacher_checkpoint_sha256),
                }
            )
    audit = {
        "status": "OK" if output else "PL_EMPTY_AFTER_SEGMENT_GATES",
        "candidate_segments": int(candidate_count),
        "selected_segments": int(len(selected)),
        "selected_frame_predictions": int(len(output)),
        "selected_segments_by_class": {str(key): int(value) for key, value in sorted(selected_by_class.items())},
        "selected_frames_by_class": {str(key): int(value) for key, value in sorted(frames_by_class.items())},
        "selected_source_videos_by_class": {
            key: sorted(values) for key, values in sorted(videos_by_class.items())
        },
        "selected_source_video_counts_by_class": {
            key: len(values) for key, values in sorted(videos_by_class.items())
        },
        "rejected_short_segments": int(rejected.get("rejected_short_segments", 0)),
        "rejected_gt_conflict_segments": int(rejected.get("rejected_gt_conflict_segments", 0)),
        "rejected_duplicate_segments": int(rejected.get("rejected_duplicate_segments", 0)),
        "rejected_segment_budget": int(rejected.get("rejected_segment_budget", 0)),
        "rejected_frame_budget": int(rejected.get("rejected_frame_budget", 0)),
        "teacher_checkpoint_sha256": str(teacher_checkpoint_sha256),
        "selected_segment_audit": [segment.as_audit_dict() for segment in selected],
        # Compatibility fields are intentionally secondary to the explicit
        # segment-level fields above.
        "accepted": int(len(output)),
        "accepted_by_class": {str(key): int(value) for key, value in sorted(frames_by_class.items())},
    }
    return output, audit
