"""Build deterministic stage views with partial labels, PL and GT replay."""

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from ..class_registry import tao_bdd_registry
from ..manifest import canonical_json_hash, read_json, write_json


STAGE_IDS = {"S0_ref": (206,), "S1_pedestrian": (206, 792), "S2_truck": (206, 792, 1122)}
NEW_IDS = {"S0_ref": (206,), "S1_pedestrian": (792,), "S2_truck": (1122,)}
OLD_IDS = {"S0_ref": (), "S1_pedestrian": (206,), "S2_truck": (206, 792)}


def _load_pl(path: Optional[str]) -> Dict[str, List[dict]]:
    result: Dict[str, List[dict]] = defaultdict(list)
    if not path:
        return result
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for pred in record.get("predictions", []):
                if int(pred.get("global_id", -1)) not in (206, 792, 1122):
                    continue
                # PL track IDs use a distinct numeric namespace from source
                # GT IDs and are never passed off as source annotations.
                pred = {
                    "track_id": 100000000 + int(pred["track_id"]),
                    "dataset_category_id": -1,
                    "global_semantic_id": int(pred["global_id"]),
                    "bbox_xyxy": [float(v) for v in pred["bbox_xyxy"]],
                    "label_source": "pl",
                    "label_status": "pseudo",
                    "score": float(pred.get("score", 0.0)),
                    "iscrowd": 0,
                }
                result[record["frame_key"]].append(pred)
    return result


def _stage_video(video: dict, stage_id: str, pl: Dict[str, List[dict]], replay_frames: Set[str], training: bool, label_scope: str) -> dict:
    active = set(STAGE_IDS[stage_id])
    new_ids = set(NEW_IDS[stage_id])
    old_ids = set(OLD_IDS[stage_id])
    output = deepcopy(video)
    output["frames"] = []
    for frame in video.get("frames", []):
        kept = []
        for ann in frame.get("annotations", []):
            gid = int(ann["global_semantic_id"])
            if not training and gid in active:
                kept.append(deepcopy(ann))
            elif gid in new_ids:
                kept.append(deepcopy(ann))
            elif gid in old_ids and frame["frame_key"] in replay_frames:
                replay = deepcopy(ann)
                replay["label_source"] = "gt_replay"
                replay["label_status"] = "reliable"
                kept.append(replay)
        if training:
            for pred in pl.get(frame["frame_key"], []):
                if int(pred["global_semantic_id"]) in old_ids:
                    kept.append(deepcopy(pred))
        frame_out = deepcopy(frame)
        frame_out["annotations"] = [a for a in kept if int(a["global_semantic_id"]) in active]
        frame_out["label_scope"] = label_scope
        supervised = set(active) if (not training or label_scope == "complete") else set(new_ids)
        supervised.update(int(a["global_semantic_id"]) for a in frame_out["annotations"] if a.get("label_source") in ("pl", "gt_replay"))
        frame_out["supervised_global_ids"] = sorted(supervised)
        output["frames"].append(frame_out)
    return output


def build_stage_view(
    canonical_path: str,
    stage_id: str,
    output_path: str,
    pl_path: Optional[str] = None,
    replay_max_frames: int = 64,
    split: str = "train",
    mode: str = "train",
    label_scope: Optional[str] = None,
) -> dict:
    manifest = read_json(canonical_path)
    pl = _load_pl(pl_path)
    videos = [v for v in manifest["videos"] if v["split"] == split]
    # Bounded replay is deterministic and independent of model scores.
    candidate_frames = sorted(
        (f["frame_key"] for v in videos for f in v.get("frames", []) if any(int(a["global_semantic_id"]) in OLD_IDS[stage_id] for a in f.get("annotations", [])))
    )
    replay_frames = set(candidate_frames[: max(0, int(replay_max_frames))])
    if mode not in ("train", "eval"):
        raise ValueError("mode must be train or eval")
    training = mode == "train"
    if label_scope is None:
        label_scope = "complete" if (not training or stage_id == "S0_ref") else "partial"
    if label_scope not in ("partial", "complete"):
        raise ValueError("label_scope must be partial or complete")
    result_videos = [_stage_video(v, stage_id, pl, replay_frames, training, label_scope) for v in videos]
    counts = Counter()
    source_counts = Counter()
    for video in result_videos:
        for frame in video["frames"]:
            for ann in frame["annotations"]:
                counts[int(ann["global_semantic_id"])] += 1
                source_counts[ann.get("label_source", "unknown")] += 1
    payload = {
        "schema_version": "cmot.v1",
        "manifest_kind": "stage_view",
        "parent_manifest": str(Path(canonical_path).name),
        "stage_id": stage_id,
        "active_global_ids": list(STAGE_IDS[stage_id]),
        "new_global_ids": list(NEW_IDS[stage_id]),
        "old_global_ids": list(OLD_IDS[stage_id]),
        "split": split,
        "mode": mode,
        "label_protocol": "naive_complete" if label_scope == "complete" else "partial_label_pl_old_replay",
        "label_scope": label_scope,
        "replay_max_frames": int(replay_max_frames),
        "replay_frame_keys": sorted(replay_frames),
        "pl_path": None if not pl_path else Path(pl_path).name,
        "stats": {
            "videos": len(result_videos),
            "frames": sum(len(v["frames"]) for v in result_videos),
            "annotations": sum(len(f["annotations"]) for v in result_videos for f in v["frames"]),
            "global_ids": dict(sorted(counts.items())),
            "label_sources": dict(sorted(source_counts.items())),
            "pl_records": sum(len(v) for v in pl.values()),
        },
        "videos": result_videos,
    }
    payload["manifest_hash"] = canonical_json_hash(payload)
    write_json(output_path, payload)
    return payload["stats"]


def export_stage_view_jsonl(view_path: str, output_path: str) -> dict:
    """Export an audit-friendly one-video-per-line view.

    The training adapter consumes the canonical JSON view.  This JSONL form
    exists for the required private manifest layout and keeps each video as a
    self-contained record while retaining the parent/stage metadata header.
    """
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
    parser.add_argument("--replay-max-frames", type=int, default=64)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--label-scope", choices=("partial", "complete"))
    parser.add_argument("--jsonl-output")
    args = parser.parse_args()
    stats = build_stage_view(args.canonical, args.stage, args.output, args.pl_jsonl, args.replay_max_frames, args.split, args.mode, args.label_scope)
    result = {"stats": stats}
    if args.jsonl_output:
        result["jsonl"] = export_stage_view_jsonl(args.output, args.jsonl_output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
