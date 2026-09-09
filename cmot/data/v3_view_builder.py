"""Build V3 immutable current/evaluation views from the canonical manifest.

This module never edits the canonical source.  Every output is a small,
hashable view with explicit protocol role and annotation provenance.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..manifest import canonical_json_hash, sha256_file, write_json


CLASS_IDS = {"car": 206, "pedestrian": 792, "truck": 1122}
ALL_IDS = (206, 792, 1122)


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = [float(v) for v in left]
    rx0, ry0, rx1, ry1 = [float(v) for v in right]
    ix0, iy0, ix1, iy1 = max(lx0, rx0), max(ly0, ry0), min(lx1, rx1), min(ly1, ry1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - inter
    return inter / union if union > 0 else 0.0


def _video_uid(video: Mapping[str, object]) -> str:
    return str(video.get("source_video_uid") or video.get("video_id"))


def _frame_uid(video_uid: str, frame: Mapping[str, object]) -> str:
    return str(frame.get("frame_uid") or "%s:frame:%06d" % (video_uid, int(frame.get("frame_index", 0))))


def _track_uid(video_uid: str, ann: Mapping[str, object]) -> str:
    return str(ann.get("track_uid") or "%s:track:%s" % (video_uid, int(ann["track_id"])))


def _stable_pl_id(identity: str) -> int:
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big") & ((1 << 62) - 1)


def _load_pl(path: Optional[str], old_ids: Sequence[int]) -> Tuple[Dict[str, List[dict]], dict, Counter]:
    result: Dict[str, List[dict]] = defaultdict(list)
    metadata: dict = {}
    stats = Counter()
    if not path:
        return result, metadata, stats
    seen_frames = set()
    allowed_old_ids = {int(value) for value in old_ids}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "metadata" or "metadata" in record:
                metadata.update(record.get("metadata", record))
                continue
            frame_key = record.get("frame_key")
            if not frame_key:
                stats["invalid_missing_frame_key"] += 1
                continue
            if frame_key in seen_frames:
                stats["invalid_duplicate_frame_key"] += 1
                continue
            seen_frames.add(frame_key)
            video_uid = str(record.get("video_uid") or record.get("video_id") or frame_key.rsplit("/", 1)[0])
            frame_index = record.get("frame_index")
            if frame_index is None:
                try:
                    frame_index = int(str(frame_key).rsplit("/", 1)[-1])
                except (TypeError, ValueError):
                    frame_index = None
            for raw in record.get("predictions", []):
                try:
                    gid = int(raw["global_id"])
                    track_id = int(raw["track_id"])
                    score = float(raw["score"])
                    bbox = [float(v) for v in raw["bbox_xyxy"]]
                except (KeyError, TypeError, ValueError):
                    stats["invalid_prediction"] += 1
                    continue
                if gid not in allowed_old_ids or not math.isfinite(score) or len(bbox) != 4:
                    stats["invalid_prediction"] += 1
                    continue
                if not all(math.isfinite(v) for v in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    stats["invalid_prediction"] += 1
                    continue
                teacher_hash = metadata.get("teacher_checkpoint_sha256")
                if not teacher_hash or str(teacher_hash) in ("unknown", "", "None"):
                    stats["missing_teacher_hash"] += 1
                    continue
                identity = "%s|%s|%d" % (video_uid, teacher_hash, track_id)
                result[str(frame_key)].append({
                    "track_id": _stable_pl_id(identity),
                    "track_uid": "%s:pl_track:%d" % (identity, _stable_pl_id(identity)),
                    "teacher_track_id": track_id,
                    "pl_identity": identity,
                    "dataset_category_id": -1,
                    "global_semantic_id": gid,
                    "bbox_xyxy": bbox,
                    "label_source": "pl",
                    "label_status": "pseudo",
                    "score": max(0.0, min(1.0, score)),
                    "iscrowd": 0,
                    "acquired_stage": "teacher_pl",
                    "teacher_checkpoint_sha256": str(teacher_hash),
                    "pl_segment_id": raw.get("pl_segment_id"),
                    "segment_length": raw.get("segment_length"),
                    "segment_reliability": raw.get("segment_reliability", raw.get("reliability")),
                    "segment_mean_score": raw.get("segment_mean_score"),
                    "reliability": raw.get("reliability"),
                    "frame_index": frame_index,
                    "timestamp_s": record.get("timestamp_s"),
                })
                stats["raw_pl"] += 1
    if not metadata.get("teacher_checkpoint_sha256"):
        stats["pl_disabled_missing_teacher_hash"] += 1
        return defaultdict(list), metadata, stats
    return result, metadata, stats


def _validate_pl_segments(
    frame_records: Mapping[str, Sequence[Mapping[str, object]]],
    metadata: Mapping[str, object],
    max_gap_s: float,
) -> None:
    """Reject malformed versioned QPL segment records before view construction."""
    policy_version = str(metadata.get("pseudo_policy_version", ""))
    if policy_version not in ("qpl_segment_v2", "qpl_segment_v3"):
        return
    grouped = defaultdict(list)
    for frame_key, predictions in frame_records.items():
        for prediction in predictions:
            segment_id = prediction.get("pl_segment_id")
            required = (segment_id, prediction.get("segment_length"), prediction.get("segment_reliability"), prediction.get("segment_mean_score"))
            if not segment_id or any(value is None for value in required[1:]):
                raise ValueError("%s prediction is missing segment metadata: %s" % (policy_version, frame_key))
            try:
                frame_index = int(prediction["frame_index"])
                segment_length = int(prediction["segment_length"])
                reliability = float(prediction["segment_reliability"])
                mean_score = float(prediction["segment_mean_score"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("%s prediction has invalid segment metadata: %s" % (policy_version, frame_key))
            if segment_length <= 0 or not math.isfinite(reliability) or not math.isfinite(mean_score):
                raise ValueError("%s prediction has non-finite segment metadata: %s" % (policy_version, frame_key))
            grouped[str(segment_id)].append((str(frame_key), frame_index, prediction))
    for segment_id, values in grouped.items():
        values.sort(key=lambda item: (item[1], item[0]))
        video_uids = {str(item[2].get("video_uid", "")) for item in values}
        global_ids = {int(item[2].get("global_semantic_id", -1)) for item in values}
        track_ids = {int(item[2].get("teacher_track_id", item[2].get("track_id", -1))) for item in values}
        if len(video_uids) != 1 or len(global_ids) != 1 or len(track_ids) != 1:
            raise ValueError("qpl segment identity changes within %s" % segment_id)
        frame_indexes = [item[1] for item in values]
        if any(right - left != 1 for left, right in zip(frame_indexes, frame_indexes[1:])):
            raise ValueError("qpl segment is not contiguous: %s" % segment_id)
        declared_lengths = {int(item[2]["segment_length"]) for item in values}
        if declared_lengths != {len(values)}:
            raise ValueError("qpl segment length mismatch: %s" % segment_id)
        for (_, _, left), (_, _, right) in zip(values, values[1:]):
            if left.get("timestamp_s") is not None and right.get("timestamp_s") is not None:
                try:
                    dt = float(right["timestamp_s"]) - float(left["timestamp_s"])
                except (TypeError, ValueError):
                    raise ValueError("qpl segment timestamp is invalid: %s" % segment_id)
                if not math.isfinite(dt) or dt <= 0.0 or dt > float(max_gap_s):
                    raise ValueError("qpl segment timestamp gap is invalid: %s" % segment_id)


def _validate_output_pl_segments(
    videos: Sequence[Mapping[str, object]], max_gap_s: float, policy_version: str
) -> None:
    grouped = defaultdict(list)
    for video in videos:
        for frame in video.get("frames", []):
            for ann in frame.get("annotations", []):
                if ann.get("label_source") != "pl":
                    continue
                value = dict(ann)
                value["video_uid"] = str(video.get("source_video_uid") or video.get("video_id"))
                value["frame_index"] = int(frame.get("frame_index", 0))
                value["frame_key"] = str(frame.get("frame_key", ""))
                value["timestamp_s"] = frame.get("timestamp_s")
                grouped[str(ann.get("pl_segment_id"))].append(value)
    _validate_pl_segments(grouped, {"pseudo_policy_version": policy_version}, max_gap_s)


def _eligible(video: Mapping[str, object], new_ids: Sequence[int]) -> bool:
    wanted = {int(value) for value in new_ids}
    return any(int(ann.get("global_semantic_id", -1)) in wanted for frame in video.get("frames", []) for ann in frame.get("annotations", []))


def deterministic_split_ids(
    canonical_path: str,
    stage: str,
    train_cap: int,
    dev_cap: int,
    calibration_cap: int,
    eval_video_ids: Sequence[str],
    seed: int = 20260907,
) -> dict:
    """Return mutually disjoint source-video partitions for one stage."""
    payload = json.loads(Path(canonical_path).read_text(encoding="utf-8"))
    stage_new = {"S0": (206,), "J3": ALL_IDS, "S1": (792,), "S2": (1122,)}[stage]
    pool = [v for v in payload.get("videos", []) if v.get("split") == "train" and _eligible(v, stage_new)]
    ranked = sorted(pool, key=lambda v: hashlib.sha256(("%s|%s|%s" % (seed, stage, v["video_id"])).encode("utf-8")).hexdigest())
    excluded = set(eval_video_ids)
    ranked = [v for v in ranked if v["video_id"] not in excluded]
    train = ranked[: int(train_cap)]
    dev = ranked[int(train_cap):int(train_cap) + int(dev_cap)]
    calibration = ranked[int(train_cap) + int(dev_cap):int(train_cap) + int(dev_cap) + int(calibration_cap)]
    return {
        "stage": stage,
        "seed": int(seed),
        "source_manifest": Path(canonical_path).name,
        "source_manifest_sha256": sha256_file(canonical_path),
        "eval_video_ids": sorted(str(v) for v in eval_video_ids),
        "train_video_ids": sorted(str(v["video_id"]) for v in train),
        "dev_video_ids": sorted(str(v["video_id"]) for v in dev),
        "calibration_video_ids": sorted(str(v["video_id"]) for v in calibration),
        "eligible_count": len(ranked),
    }


def write_split_manifest(path: str, split_payload: dict) -> dict:
    value = dict(split_payload)
    value["manifest_hash"] = canonical_json_hash(value)
    write_json(path, value)
    return value


def _current_annotations(frame: Mapping[str, object], ids: Sequence[int]) -> List[dict]:
    wanted = {int(value) for value in ids}
    result = []
    for raw in frame.get("annotations", []):
        if int(raw.get("global_semantic_id", -1)) not in wanted:
            continue
        if int(raw.get("iscrowd", 0)) or raw.get("label_status") == "ignore":
            continue
        ann = deepcopy(raw)
        ann["label_source"] = "gt"
        ann["label_status"] = "reliable"
        ann["acquired_stage"] = "source_gt"
        ann["teacher_checkpoint_sha256"] = None
        ann["pl_segment_id"] = None
        ann["reliability"] = 1.0
        result.append(ann)
    return result


def build_view(
    canonical_path: str,
    output_path: str,
    stage: str,
    mode: str,
    protocol_role: str,
    video_ids: Sequence[str],
    active_ids: Sequence[int],
    new_ids: Sequence[int],
    old_ids: Sequence[int] = (),
    pl_path: Optional[str] = None,
    conflict_iou: float = 0.7,
    duplicate_iou: float = 0.7,
    split: str = "train",
    output_split: Optional[str] = None,
    enable_pl: bool = False,
    total_pl_cap: Optional[int] = None,
    min_segment_frames: int = 3,
    max_gap_s: float = 1.0,
) -> dict:
    """Build a view for one fixed source-video partition."""
    canonical = json.loads(Path(canonical_path).read_text(encoding="utf-8"))
    selected = set(str(v) for v in video_ids)
    pl, pl_metadata, pl_stats = _load_pl(pl_path if enable_pl else None, old_ids)
    stage_stats = Counter(pl_stats)
    qpl_policy_version = str(pl_metadata.get("pseudo_policy_version", ""))
    qpl_segment_mode = bool(enable_pl and qpl_policy_version in ("qpl_segment_v2", "qpl_segment_v3"))
    if qpl_segment_mode:
        _validate_pl_segments(pl, pl_metadata, max_gap_s)
    if enable_pl and pl_path and not pl_metadata.get("teacher_checkpoint_sha256"):
        enable_pl = False
        stage_stats["pl_disabled_missing_teacher_hash"] += 1
    output_videos = []
    total_pl = 0
    for video in canonical.get("videos", []):
        if str(video.get("video_id")) not in selected or str(video.get("split")) != str(split):
            continue
        video_uid = _video_uid(video)
        new_video = deepcopy(video)
        new_video["source_video_uid"] = video_uid
        new_video["stream"] = "current" if mode == "train" else "eval"
        new_video["protocol_role"] = protocol_role
        new_video["frames"] = []
        for raw_frame in video.get("frames", []):
            frame = deepcopy(raw_frame)
            frame_uid = _frame_uid(video_uid, frame)
            frame["source_video_uid"] = video_uid
            frame["frame_uid"] = frame_uid
            for ann in frame.get("annotations", []):
                ann.setdefault("track_uid", _track_uid(video_uid, ann))
            if mode == "eval" or protocol_role == "diagnostic_joint":
                kept = _current_annotations(frame, active_ids)
                exhaustive = list(int(v) for v in active_ids)
                protocol = "immutable_eval_gt" if mode == "eval" else "diagnostic_joint_gt"
            else:
                kept = _current_annotations(frame, new_ids)
                exhaustive = list(int(v) for v in new_ids)
                protocol = "current_new_gt_only"
                candidates = []
                if enable_pl:
                    for pred in sorted(pl.get(str(frame.get("frame_key")), []), key=lambda x: (-float(x["score"]), x["pl_identity"])):
                        # Versioned qpl_segment policies have already applied
                        # their only budget at
                        # segment level.  Legacy V2/V3 snapshots retain the
                        # old frame cap solely for backwards compatibility.
                        if not qpl_segment_mode and total_pl_cap is not None and total_pl >= int(total_pl_cap):
                            break
                        if int(pred["global_semantic_id"]) not in set(int(v) for v in old_ids):
                            continue
                        has_conflict = any(
                            int(gt["global_semantic_id"]) != int(pred["global_semantic_id"])
                            and _bbox_iou(gt["bbox_xyxy"], pred["bbox_xyxy"]) >= float(conflict_iou)
                            for gt in kept
                        )
                        has_duplicate = any(
                            int(other["global_semantic_id"]) == int(pred["global_semantic_id"])
                            and _bbox_iou(other["bbox_xyxy"], pred["bbox_xyxy"]) >= float(duplicate_iou)
                            for other in candidates
                        )
                        if qpl_segment_mode and (has_conflict or has_duplicate):
                            # A prevalidated segment cannot lose a frame here:
                            # doing so would silently violate its continuity.
                            reason = "GT conflict" if has_conflict else "duplicate"
                            raise ValueError("%s %s at frame %s" % (qpl_policy_version, reason, frame.get("frame_key")))
                        if has_conflict:
                            stage_stats["gt_pl_conflict"] += 1
                            continue
                        if has_duplicate:
                            stage_stats["pl_duplicate"] += 1
                            continue
                        candidates.append(deepcopy(pred))
                    kept.extend(candidates)
                    total_pl += len(candidates)
                if candidates:
                    protocol = "current_new_gt_plus_quality_pl"
            if not bool(frame.get("annotation_valid", True)):
                exhaustive = []
            frame["annotations"] = kept
            frame["label_scope"] = "complete" if mode == "eval" or protocol_role == "diagnostic_joint" else "partial"
            frame["exhaustive_global_ids"] = sorted(set(exhaustive))
            frame["supervised_global_ids"] = list(frame["exhaustive_global_ids"])
            frame["annotation_valid"] = bool(frame.get("annotation_valid", True))
            frame["supervision_protocol"] = protocol
            new_video["frames"].append(frame)
        if new_video["frames"]:
            if output_split is not None:
                new_video["split"] = str(output_split)
            output_videos.append(new_video)
    if qpl_segment_mode:
        _validate_output_pl_segments(output_videos, max_gap_s, qpl_policy_version)
    result = {
        "schema_version": "cmot.continual_v3.view",
        "view_kind": "current" if mode == "train" else "evaluation",
        "stage": stage,
        "mode": mode,
        "protocol_role": protocol_role,
        "active_global_ids": [int(v) for v in active_ids],
        "new_global_ids": [int(v) for v in new_ids],
        "old_global_ids": [int(v) for v in old_ids],
        "source_manifest": Path(canonical_path).name,
        "source_manifest_sha256": sha256_file(canonical_path),
        "pl_source": None if not pl_path else Path(pl_path).name,
        "pl_metadata": {
            k: v
            for k, v in pl_metadata.items()
            if k in (
                "pseudo_policy_version",
                "pseudo_policy_hash",
                "teacher_checkpoint_sha256",
                "raw_prediction_sha256",
                "calibration_view_sha256",
                "prediction_sha256",
                "frame_list_sha256",
            )
        },
        "stats": {
            "videos": len(output_videos),
            "frames": sum(len(v["frames"]) for v in output_videos),
            "annotations": sum(len(f.get("annotations", [])) for v in output_videos for f in v["frames"]),
            "annotations_by_source": dict(Counter(a.get("label_source", "unknown") for v in output_videos for f in v["frames"] for a in f.get("annotations", []))),
            "pl_used": int(total_pl),
            "qpl_segment_mode": bool(qpl_segment_mode),
            "stage_events": dict(sorted(stage_stats.items())),
        },
        "videos": output_videos,
    }
    result["manifest_hash"] = canonical_json_hash(result)
    write_json(output_path, result)
    return result
