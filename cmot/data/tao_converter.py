"""Convert the locally available TAO-Amodal BDD subset to C-MOT schema."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ..class_registry import tao_bdd_registry
from ..manifest import canonical_json_hash, write_json
from ..schema import AnnotationRecord, FrameRecord, VideoRecord


TAO_CATEGORY_MAP = {211: "car", 805: "pedestrian", 1144: "truck"}
GLOBAL_IDS = {"car": 206, "pedestrian": 792, "truck": 1122}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _convert_split(annotation_path: Path, split: str, frames_root: Path) -> tuple:
    source = _load_json(annotation_path)
    videos_by_id = {
        int(v["id"]): v
        for v in source.get("videos", [])
        if v.get("metadata", {}).get("dataset") == "BDD"
    }
    images_by_video: Dict[int, List[dict]] = defaultdict(list)
    images_by_id: Dict[int, dict] = {}
    for image in source.get("images", []):
        video_id = image.get("video_id")
        if video_id is None or int(video_id) not in videos_by_id:
            continue
        images_by_video[int(video_id)].append(image)
        images_by_id[int(image["id"])] = image
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in source.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        if image_id in images_by_id and int(ann.get("category_id", -1)) in TAO_CATEGORY_MAP:
            anns_by_image[image_id].append(ann)

    videos: List[VideoRecord] = []
    missing_images: List[str] = []
    annotation_counts = Counter()
    for source_video_id in sorted(videos_by_id):
        video = videos_by_id[source_video_id]
        video_name = str(video["name"])
        video_id = "%s_bdd_%04d" % (split, source_video_id)
        frames: List[FrameRecord] = []
        for image in sorted(images_by_video[source_video_id], key=lambda x: (int(x.get("frame_index", 0)), int(x["id"]))):
            rel_file = str(image["file_name"])
            if not (frames_root / rel_file).is_file():
                missing_images.append(rel_file)
            frame_index = int(image.get("frame_index", 0))
            frame_key = "%s/%06d" % (video_id, frame_index)
            converted: List[AnnotationRecord] = []
            for ann in sorted(anns_by_image.get(int(image["id"]), []), key=lambda x: (int(x.get("track_id", -1)), int(x["id"]))):
                x, y, w, h = [float(v) for v in ann["bbox"]]
                if w <= 0 or h <= 0:
                    continue
                category_id = int(ann["category_id"])
                name = TAO_CATEGORY_MAP[category_id]
                converted.append(
                    AnnotationRecord(
                        track_id=int(ann.get("track_id", ann["id"])),
                        dataset_category_id=category_id,
                        global_semantic_id=GLOBAL_IDS[name],
                        bbox_xyxy=[x, y, x + w, y + h],
                        label_source="gt",
                        label_status="reliable" if not int(ann.get("iscrowd", 0)) else "ignore",
                        iscrowd=int(ann.get("iscrowd", 0)),
                    )
                )
                annotation_counts[name] += 1
            frames.append(
                FrameRecord(
                    frame_key=frame_key,
                    frame_index=frame_index,
                    file_name=rel_file,
                    width=int(image["width"]),
                    height=int(image["height"]),
                    annotations=converted,
                    timestamp_s=float(frame_index) / 30.0,
                    label_scope="partial" if category_id_in_not_exhaustive(video, TAO_CATEGORY_MAP) else "complete",
                    supervised_global_ids=list(GLOBAL_IDS.values()),
                    exhaustive_global_ids=list(GLOBAL_IDS.values()),
                    annotation_valid=True,
                    source_image_id=int(image["id"]),
                )
            )
        if frames:
            videos.append(
                VideoRecord(
                    video_id=video_id,
                    split=split,
                    width=int(video["width"]),
                    height=int(video["height"]),
                    frames=frames,
                    source_video_id=source_video_id,
                    source_video_name=video_name,
                    dataset="TAO-Amodal/BDD",
                    image_root="frames",
                )
            )
    return videos, missing_images, annotation_counts


def category_id_in_not_exhaustive(video: Mapping[str, object], category_map: Mapping[int, str]) -> bool:
    not_exhaustive = {int(v) for v in video.get("not_exhaustive_category_ids", [])}
    return bool(not_exhaustive.intersection(category_map.keys()))


def convert_tao_bdd(
    tao_root: str,
    output_path: str,
    require_images: bool = True,
) -> dict:
    root = Path(tao_root)
    frames_root = root / "frames"
    all_videos: List[VideoRecord] = []
    missing: List[str] = []
    counts = Counter()
    split_stats = {}
    for split, annotation_name in (("train", "train.json"), ("val", "validation.json")):
        videos, split_missing, split_counts = _convert_split(root / "annotations" / annotation_name, split, frames_root)
        all_videos.extend(videos)
        missing.extend(split_missing)
        counts.update(split_counts)
        split_stats[split] = {
            "videos": len(videos),
            "frames": sum(len(v.frames) for v in videos),
            "annotations": sum(len(f.annotations) for v in videos for f in v.frames),
            "class_annotations": dict(sorted(split_counts.items())),
        }
    if require_images and missing:
        raise FileNotFoundError("%d TAO-BDD frame files are missing; first=%s" % (len(missing), missing[0]))
    registry = tao_bdd_registry()
    payload = {
        "schema_version": "cmot.v1",
        "manifest_kind": "canonical_real_video",
        "source": {
            "dataset": "TAO-Amodal/BDD",
            "annotation_files": ["annotations/train.json", "annotations/validation.json"],
            "image_root": "frames",
            "source_category_id_map": {str(k): v for k, v in sorted(TAO_CATEGORY_MAP.items())},
            "semantic_id_map": {k: int(v) for k, v in GLOBAL_IDS.items()},
            "note": "BDD frames and TAO sparse annotations; TAO category 805 has synset person.n.01 and display name baby.",
        },
        "id_namespaces": {
            "dataset_category_id": "TAO/LVIS category ID",
            "global_semantic_id": "zero-based row in the 1203-row LVIS semantic bank",
            "text_row": "same semantic-bank row, recorded in registry",
            "select_column_id": "position inside a stage active_global_ids tuple",
            "track_id": "source track ID; PL IDs are explicitly namespaced by the view builder",
        },
        "registry": registry.as_dict(),
        "stats": {
            "videos": len(all_videos),
            "frames": sum(len(v.frames) for v in all_videos),
            "annotations": sum(len(f.annotations) for v in all_videos for f in v.frames),
            "class_annotations": dict(sorted(counts.items())),
            "split": split_stats,
            "missing_images": len(missing),
        },
        "videos": [v.as_dict() for v in all_videos],
    }
    payload["manifest_hash"] = canonical_json_hash(payload)
    write_json(output_path, payload)
    return payload["stats"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tao-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()
    stats = convert_tao_bdd(args.tao_root, args.output, require_images=not args.allow_missing_images)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
