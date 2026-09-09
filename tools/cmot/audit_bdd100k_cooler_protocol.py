"""Audit the full-data gate for the COOLer BDD100K M→L protocol.

The command is read-only.  It resolves data only from the supplied runtime
binding; it does not search the whole machine, download data, or fall back to
the parent/Codex proxy.  A 190-video or other partial manifest is reported as
blocked rather than being silently accepted as a benchmark.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cmot.manifest import sha256_file, write_json


EXPECTED_FOUNDATION_SHA256 = "0862cac87ad50f58a01ce17d4e44af0468ad8639cfccd18d66d2e9b2570d839e"
EXPECTED = {"train": 1400, "val": 200}


def _runtime(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("paths"), dict):
        raise ValueError("runtime JSON must contain a paths object")
    return value


def _source_path(paths: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = paths.get(key)
        if value:
            return str(value)
    return None


def _count_cmot(payload: Mapping[str, Any]) -> Tuple[dict, dict]:
    counts = {split: {"videos": 0, "frames": 0, "annotations": 0, "annotation_videos": 0} for split in EXPECTED}
    for video in payload.get("videos", []):
        split = str(video.get("split", ""))
        if split not in counts:
            continue
        rows = [ann for frame in video.get("frames", []) for ann in frame.get("annotations", [])]
        counts[split]["videos"] += 1
        counts[split]["frames"] += len(video.get("frames", []))
        counts[split]["annotations"] += len(rows)
        if rows:
            counts[split]["annotation_videos"] += 1
    return counts, {"kind": "cmot_manifest", "manifest_hash": payload.get("manifest_hash")}


def _count_coco(payload: Mapping[str, Any]) -> Tuple[dict, dict]:
    videos = {int(item["id"]): dict(item) for item in payload.get("videos", []) if "id" in item}
    images_by_video = defaultdict(list)
    for image in payload.get("images", []):
        if "video_id" in image:
            images_by_video[int(image["video_id"])].append(image)
    anns_by_image = defaultdict(list)
    for ann in payload.get("annotations", []):
        if "image_id" in ann:
            anns_by_image[int(ann["image_id"])].append(ann)
    counts = {split: {"videos": 0, "frames": 0, "annotations": 0, "annotation_videos": 0} for split in EXPECTED}
    missing_split = False
    for video_id, video in videos.items():
        split = video.get("split")
        if split not in EXPECTED:
            missing_split = True
            continue
        frames = images_by_video.get(video_id, [])
        annotations = [ann for image in frames for ann in anns_by_image.get(int(image.get("id", -1)), [])]
        counts[split]["videos"] += 1
        counts[split]["frames"] += len(frames)
        counts[split]["annotations"] += len(annotations)
        if annotations:
            counts[split]["annotation_videos"] += 1
    return counts, {"kind": "coco_box_track_20", "videos_without_train_val_split": missing_split}


def _count_source(path: str) -> Tuple[dict, dict, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("BDD source must be a JSON object")
    if "images" in payload and "annotations" in payload:
        counts, kind = _count_coco(payload)
    else:
        counts, kind = _count_cmot(payload)
    return counts, kind, payload


def _image_audit(image_root: Optional[str], payload: Mapping[str, Any]) -> dict:
    result = {"root_present": bool(image_root and Path(image_root).is_dir()), "referenced_frames": 0, "missing_referenced_frames": 0}
    if not image_root or not Path(image_root).is_dir() or "videos" not in payload:
        return result
    root = Path(image_root)
    for video in payload.get("videos", []):
        video_root = video.get("image_root")
        for frame in video.get("frames", []):
            result["referenced_frames"] += 1
            candidate = root / str(video_root) / str(frame.get("file_name")) if video_root else root / str(frame.get("file_name"))
            if not candidate.is_file():
                result["missing_referenced_frames"] += 1
    return result


def _artifact_audit(paths: Mapping[str, Any]) -> dict:
    names = {}
    for key in ("foundation_checkpoint", "text_embedding", "image_embedding", "config_file", "trackeval_root"):
        value = paths.get(key)
        if not value:
            names[key] = {"basename": None, "exists": False}
            continue
        path = Path(str(value))
        item = {"basename": path.name, "exists": path.exists(), "is_file": path.is_file()}
        if path.is_file():
            item["bytes"] = int(path.stat().st_size)
            if key in ("foundation_checkpoint", "text_embedding", "image_embedding"):
                item["sha256"] = sha256_file(str(path))
        names[key] = item
    foundation = names.get("foundation_checkpoint", {})
    names["foundation_hash_match"] = foundation.get("sha256") == EXPECTED_FOUNDATION_SHA256
    names["credential_values_recorded"] = False
    names["proxy_values_recorded"] = False
    return names


def audit_bdd100k_cooler_protocol(runtime_path: str, output_path: Optional[str] = None) -> dict:
    runtime = _runtime(runtime_path)
    paths = runtime["paths"]
    source = _source_path(paths, "bdd_manifest", "raw_bdd_manifest", "canonical_manifest")
    if not source or not Path(source).is_file():
        result = {
            "schema_version": "cmot.cooler_compat.data_audit.v1",
            "status": "BLOCKED_FULL_BDD_MISSING",
            "blockers": ["BDD_SOURCE_MANIFEST_MISSING"],
            "train_video_count": 0,
            "val_video_count": 0,
            "train_frame_count": 0,
            "val_frame_count": 0,
            "annotation_video_count_train": 0,
            "annotation_video_count_val": 0,
            "downloads": "NOT_RUN_no_new_data_or_dependency_download",
        }
        if output_path:
            write_json(output_path, result)
        return result
    counts, source_info, payload = _count_source(source)
    image_root = _source_path(paths, "bdd_image_root", "raw_bdd_image_root", "image_root")
    train_count = int(counts["train"]["videos"])
    val_count = int(counts["val"]["videos"])
    blockers = []
    if train_count != EXPECTED["train"] or val_count != EXPECTED["val"]:
        blockers.append("BLOCKED_FULL_BDD_MISSING")
    if counts["train"]["annotation_videos"] != EXPECTED["train"] or counts["val"]["annotation_videos"] != EXPECTED["val"]:
        blockers.append("BDD_ANNOTATION_VIDEO_COUNT_MISMATCH")
    image_audit = _image_audit(image_root, payload)
    if image_audit["missing_referenced_frames"]:
        blockers.append("BDD_REFERENCED_IMAGE_MISSING")
    artifacts = _artifact_audit(paths)
    if not artifacts.get("foundation_hash_match"):
        blockers.append("BLOCKED_FOUNDATION_HASH_MISMATCH")
    result = {
        "schema_version": "cmot.cooler_compat.data_audit.v1",
        "status": "OK" if not blockers else ("BLOCKED_FOUNDATION_HASH_MISMATCH" if "BLOCKED_FOUNDATION_HASH_MISMATCH" in blockers and len(blockers) == 1 else "BLOCKED_FULL_BDD_MISSING"),
        "blockers": sorted(set(blockers)),
        "train_video_count": train_count,
        "val_video_count": val_count,
        "train_frame_count": int(counts["train"]["frames"]),
        "val_frame_count": int(counts["val"]["frames"]),
        "annotation_video_count_train": int(counts["train"]["annotation_videos"]),
        "annotation_video_count_val": int(counts["val"]["annotation_videos"]),
        "train_annotation_count": int(counts["train"]["annotations"]),
        "val_annotation_count": int(counts["val"]["annotations"]),
        "expected": {"train_videos": 1400, "val_videos": 200},
        "source": {"basename": Path(source).name, "sha256": sha256_file(source), **source_info},
        "image_audit": image_audit,
        "artifact_audit": artifacts,
        "runtime_basename": Path(runtime_path).name,
        "downloads": "NOT_RUN_no_new_data_or_dependency_download",
        "full_bdd_gate_passed": not blockers,
    }
    if output_path:
        write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = audit_bdd100k_cooler_protocol(args.runtime, args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value.get("status") == "OK" else 2)


if __name__ == "__main__":
    main()
