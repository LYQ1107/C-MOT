"""Build cmot.v2 current, replay and immutable evaluation views.

The canonical manifest is source material only.  A repair_v2 training view is
constructed from legal new-class GT, teacher PL, and (when requested) a
separately saved replay memory.  Hidden old/future GT never enters a current
view.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..manifest import canonical_json_hash, read_json, write_json


STAGE_IDS = {"S0_ref": (206,), "S1_pedestrian": (206, 792), "S2_truck": (206, 792, 1122)}
NEW_IDS = {"S0_ref": (206,), "S1_pedestrian": (792,), "S2_truck": (1122,)}
OLD_IDS = {"S0_ref": (), "S1_pedestrian": (206,), "S2_truck": (206, 792)}


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = [float(v) for v in left]
    rx0, ry0, rx1, ry1 = [float(v) for v in right]
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    la = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    ra = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = la + ra - inter
    return 0.0 if union <= 0 else inter / union


def _stable_pl_id(video_uid: str, teacher_hash: str, track_id: int) -> int:
    token = "%s|%s|%s" % (video_uid, teacher_hash, int(track_id))
    # Keep the value in signed int64 while leaving source GT IDs untouched.
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") & ((1 << 62) - 1)


def _frame_exhaustive(frame: dict) -> List[int]:
    if "exhaustive_global_ids" in frame:
        return sorted({int(value) for value in frame.get("exhaustive_global_ids", [])})
    # Only canonical v1 source frames are interpreted this way.  Ambiguous
    # v1 stage views are rejected by build_stage_view below.
    if frame.get("label_scope") == "complete":
        return sorted({int(value) for value in frame.get("supervised_global_ids", [])})
    return []


def _ignore_regions(frame: dict) -> List[dict]:
    regions = [deepcopy(value) for value in frame.get("ignore_regions", [])]
    for ann in frame.get("annotations", []):
        if int(ann.get("iscrowd", 0)) or ann.get("label_status") == "ignore":
            regions.append({
                "bbox_xyxy": [float(v) for v in ann["bbox_xyxy"]],
                "global_semantic_id": int(ann.get("global_semantic_id", -1)),
                "iscrowd": int(ann.get("iscrowd", 0)),
                "source": "raw_ignore",
            })
    return regions


def _load_pl(
    path: Optional[str],
    score_threshold: float = 0.19,
) -> Tuple[Dict[str, List[dict]], dict, dict]:
    """Load and validate a teacher JSONL without reading hidden GT."""
    result: Dict[str, List[dict]] = defaultdict(list)
    metadata: dict = {}
    stats = Counter()
    if not path:
        return result, metadata, dict(stats)
    seen_frames: Set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "metadata" or "metadata" in record:
                metadata.update(record.get("metadata", record))
                continue
            frame_key = record.get("frame_key")
            if not frame_key:
                raise ValueError("PL line %d has no frame_key" % line_number)
            if frame_key in seen_frames:
                raise ValueError("duplicate PL frame_key: %s" % frame_key)
            seen_frames.add(frame_key)
            video_uid = str(record.get("video_uid") or frame_key.rsplit("/", 1)[0])
            teacher_hash = str(metadata.get("teacher_checkpoint_sha256") or metadata.get("teacher_hash") or "unknown")
            for raw in record.get("predictions", []):
                gid = int(raw.get("global_id", -1))
                score = raw.get("score")
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = float("nan")
                bbox = raw.get("bbox_xyxy")
                if gid not in (206, 792, 1122) or bbox is None or len(bbox) != 4 or not math.isfinite(score):
                    stats["invalid_pl"] += 1
                    continue
                if score < float(score_threshold):
                    stats["below_threshold"] += 1
                    continue
                values = [float(v) for v in bbox]
                if not all(math.isfinite(v) for v in values) or values[2] <= values[0] or values[3] <= values[1]:
                    stats["invalid_pl"] += 1
                    continue
                original_track_id = int(raw["track_id"])
                result[frame_key].append({
                    "track_id": _stable_pl_id(video_uid, teacher_hash, original_track_id),
                    "teacher_track_id": original_track_id,
                    "pl_identity": "%s|%s|%s" % (video_uid, teacher_hash, original_track_id),
                    "dataset_category_id": -1,
                    "global_semantic_id": gid,
                    "bbox_xyxy": values,
                    "label_source": "pl",
                    "label_status": "pseudo",
                    "score": max(0.0, min(1.0, score)),
                    "iscrowd": 0,
                })
                stats["kept_pl"] += 1
    metadata.setdefault("pl_score_threshold", float(score_threshold))
    metadata["frame_list_sha256"] = canonical_json_hash(sorted(seen_frames))
    stats["pl_frames"] = len(seen_frames)
    return result, metadata, dict(stats)


def _select_videos(videos: List[dict], stage_id: str, cap: Optional[int], seed: int) -> List[dict]:
    if cap is None or len(videos) <= int(cap):
        return videos
    new_ids = set(NEW_IDS[stage_id])
    eligible = [v for v in videos if any(
        any(int(a.get("global_semantic_id", -1)) in new_ids for a in f.get("annotations", []))
        for f in v.get("frames", [])
    )]
    pool = eligible or videos
    ranked = sorted(
        pool,
        key=lambda value: hashlib.sha256(("%s|%s|%s" % (seed, stage_id, value["video_id"])).encode("utf-8")).hexdigest(),
    )
    chosen = ranked[: int(cap)]
    if len(chosen) < int(cap):
        seen = {v["video_id"] for v in chosen}
        chosen.extend(v for v in sorted(videos, key=lambda x: x["video_id"]) if v["video_id"] not in seen)
        chosen = chosen[: int(cap)]
    return sorted(chosen, key=lambda value: value["video_id"])


def _deduplicate_pl(predictions: List[dict], duplicate_iou: float, stats: Counter) -> List[dict]:
    kept: List[dict] = []
    for pred in sorted(predictions, key=lambda value: (-float(value["score"]), str(value["pl_identity"]))):
        duplicate = any(
            int(other["global_semantic_id"]) == int(pred["global_semantic_id"])
            and _bbox_iou(other["bbox_xyxy"], pred["bbox_xyxy"]) >= float(duplicate_iou)
            for other in kept
        )
        if duplicate:
            stats["duplicate_pl"] += 1
        else:
            kept.append(pred)
    return kept


def _stage_video(video: dict, stage_id: str, pl: Dict[str, List[dict]], training: bool, label_scope: str, stats: Counter, conflict_iou: float, pl_duplicate_iou: float) -> dict:
    active = set(STAGE_IDS[stage_id])
    new_ids = set(NEW_IDS[stage_id])
    old_ids = set(OLD_IDS[stage_id])
    output = deepcopy(video)
    output["frames"] = []
    output["stream"] = "current" if training else "eval"
    for frame in video.get("frames", []):
        raw_annotations = [
            deepcopy(ann) for ann in frame.get("annotations", [])
            if int(ann.get("global_semantic_id", -1)) in active and not int(ann.get("iscrowd", 0))
            and ann.get("label_status") != "ignore"
        ]
        current_gt = [ann for ann in raw_annotations if int(ann["global_semantic_id"]) in new_ids]
        if training:
            kept = list(current_gt)
            candidates = []
            for pred in pl.get(frame["frame_key"], []):
                if int(pred["global_semantic_id"]) not in old_ids:
                    continue
                conflict = any(
                    int(gt["global_semantic_id"]) != int(pred["global_semantic_id"])
                    and _bbox_iou(gt["bbox_xyxy"], pred["bbox_xyxy"]) >= float(conflict_iou)
                    for gt in current_gt
                )
                if conflict:
                    stats["gt_pl_conflict"] += 1
                    continue
                candidates.append(pred)
            kept.extend(_deduplicate_pl(candidates, float(pl_duplicate_iou), stats))
            exhaustive = [gid for gid in _frame_exhaustive(frame) if gid in new_ids]
            protocol = "current_new_gt_plus_old_pl"
            effective_scope = label_scope
        else:
            kept = raw_annotations
            exhaustive = [gid for gid in _frame_exhaustive(frame) if gid in active]
            protocol = "immutable_eval_gt"
            effective_scope = "complete"
        frame_out = deepcopy(frame)
        frame_out["annotations"] = kept
        frame_out["label_scope"] = effective_scope
        frame_out["exhaustive_global_ids"] = sorted(set(int(v) for v in exhaustive))
        frame_out["supervised_global_ids"] = list(frame_out["exhaustive_global_ids"])
        frame_out["ignore_regions"] = _ignore_regions(frame)
        frame_out["annotation_valid"] = bool(frame.get("annotation_valid", True))
        frame_out["supervision_protocol"] = protocol
        output["frames"].append(frame_out)
    return output


def build_stage_view(
    canonical_path: str,
    stage_id: str,
    output_path: str,
    pl_path: Optional[str] = None,
    replay_max_frames: Optional[int] = None,
    split: str = "train",
    mode: str = "train",
    label_scope: Optional[str] = None,
    *,
    memory_path: Optional[str] = None,
    replay_output_path: Optional[str] = None,
    pl_score_threshold: float = 0.19,
    conflict_iou: float = 0.9,
    pl_duplicate_iou: float = 0.9,
    train_source_cap: Optional[int] = None,
    video_ids: Optional[Sequence[str]] = None,
    seed: int = 20260907,
) -> dict:
    if replay_max_frames is not None:
        raise ValueError("replay_max_frames is deprecated; provide memory_path and replay_output_path")
    if stage_id not in STAGE_IDS:
        raise ValueError("unknown stage_id %s" % stage_id)
    if mode not in ("train", "eval"):
        raise ValueError("mode must be train or eval")
    manifest = read_json(canonical_path)
    if manifest.get("manifest_kind") != "canonical_real_video":
        raise ValueError("repair_v2 views must be rebuilt from canonical_real_video, not a v1 stage view")
    pl, pl_metadata, pl_stats = _load_pl(pl_path, score_threshold=pl_score_threshold)
    training = mode == "train"
    if label_scope is None:
        label_scope = "partial" if training and stage_id != "S0_ref" else "complete"
    if training and label_scope != "partial" and stage_id != "S0_ref":
        raise ValueError("repair_v2 incremental current views require partial label_scope")
    selected = [v for v in manifest.get("videos", []) if v.get("split") == split]
    if video_ids:
        wanted = {str(value) for value in video_ids}
        selected = [v for v in selected if str(v["video_id"]) in wanted]
    elif training:
        selected = _select_videos(selected, stage_id, train_source_cap, seed)
    else:
        selected = sorted(selected, key=lambda value: value["video_id"])
    stats = Counter()
    result_videos = [_stage_video(v, stage_id, pl, training, label_scope, stats, conflict_iou, pl_duplicate_iou) for v in selected]
    counts = Counter()
    source_counts = Counter()
    exhaustive_counts = Counter()
    for video in result_videos:
        for frame in video["frames"]:
            exhaustive_counts.update(frame.get("exhaustive_global_ids", []))
            for ann in frame["annotations"]:
                counts[int(ann["global_semantic_id"])] += 1
                source_counts[ann.get("label_source", "unknown")] += 1
    payload = {
        "schema_version": "cmot.v2",
        "manifest_kind": "stage_view",
        "parent_manifest": Path(canonical_path).name,
        "stage_id": stage_id,
        "active_global_ids": list(STAGE_IDS[stage_id]),
        "new_global_ids": list(NEW_IDS[stage_id]),
        "old_global_ids": list(OLD_IDS[stage_id]),
        "split": split,
        "mode": mode,
        "stream": "current" if training else "eval",
        "label_protocol": "current_new_gt_plus_old_pl" if training and stage_id != "S0_ref" else ("current_gt" if training else "immutable_eval_gt"),
        "label_scope": label_scope,
        "pl_metadata": pl_metadata,
        "pl_path": None if not pl_path else Path(pl_path).name,
        "memory_path": None if not memory_path else Path(memory_path).name,
        "stats": {
            "videos": len(result_videos),
            "frames": sum(len(v["frames"]) for v in result_videos),
            "annotations": sum(len(f["annotations"]) for v in result_videos for f in v["frames"]),
            "global_ids": dict(sorted(counts.items())),
            "exhaustive_global_ids": dict(sorted(exhaustive_counts.items())),
            "label_sources": dict(sorted(source_counts.items())),
            **dict(sorted(stats.items())),
            **{("pl_" + str(key)): value for key, value in sorted(pl_stats.items())},
        },
        "videos": result_videos,
    }
    payload["manifest_hash"] = canonical_json_hash(payload)
    write_json(output_path, payload)

    if replay_output_path:
        if not memory_path:
            raise ValueError("replay_output_path requires memory_path")
        from ..memory.clip_memory import ClipReplayMemory
        memory_payload = read_json(memory_path)
        memory = ClipReplayMemory.from_payload(memory_payload)
        replay = memory.as_view(stage_id=stage_id, active_global_ids=OLD_IDS[stage_id], split=split)
        write_json(replay_output_path, replay)
        payload["replay_output"] = Path(replay_output_path).name
    return payload["stats"]


def export_stage_view_jsonl(view_path: str, output_path: str) -> dict:
    payload = read_json(view_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = {key: value for key, value in payload.items() if key != "videos"}
    header["record_type"] = "manifest"
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True, ensure_ascii=False) + "\n")
        for video in payload.get("videos", []):
            handle.write(json.dumps({"record_type": "video", "video": video}, sort_keys=True, ensure_ascii=False) + "\n")
    return {"videos": len(payload.get("videos", [])), "manifest_hash": payload.get("manifest_hash"), "output": destination.name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_IDS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pl-jsonl")
    parser.add_argument("--memory-path")
    parser.add_argument("--replay-output")
    parser.add_argument("--pl-score-threshold", type=float, default=0.19)
    parser.add_argument("--conflict-iou", type=float, default=0.9)
    parser.add_argument("--pl-duplicate-iou", type=float, default=0.9)
    parser.add_argument("--train-source-cap", type=int)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--label-scope", choices=("partial", "complete"))
    parser.add_argument("--jsonl-output")
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    stats = build_stage_view(
        args.canonical, args.stage, args.output, args.pl_jsonl, None, args.split, args.mode, args.label_scope,
        memory_path=args.memory_path, replay_output_path=args.replay_output,
        pl_score_threshold=args.pl_score_threshold, conflict_iou=args.conflict_iou,
        pl_duplicate_iou=args.pl_duplicate_iou, train_source_cap=args.train_source_cap,
        video_ids=args.video_id or None, seed=args.seed,
    )
    result = {"stats": stats}
    if args.jsonl_output:
        result["jsonl"] = export_stage_view_jsonl(args.output, args.jsonl_output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
