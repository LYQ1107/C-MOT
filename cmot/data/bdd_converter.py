"""Convert locally staged BDD100K box-track COCO files to C-MOT schema."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ..class_registry import tao_bdd_registry
from ..manifest import canonical_json_hash, write_json
from ..schema import AnnotationRecord, FrameRecord, VideoRecord


BDD_CATEGORY_MAP = {1: "pedestrian", 2: "car", 3: "truck"}
GLOBAL_IDS = {"car": 206, "pedestrian": 792, "truck": 1122}


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _convert_file(annotation_path: Path, image_root: Path, split: str, domain: str) -> Tuple[List[VideoRecord], List[str], Counter]:
    source = _read(annotation_path)
    videos = {int(v["id"]): v for v in source.get("videos", [])}
    images_by_id = {int(i["id"]): i for i in source.get("images", [])}
    images_by_video: Dict[int, List[dict]] = defaultdict(list)
    for image in images_by_id.values():
        images_by_video[int(image["video_id"])].append(image)
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in source.get("annotations", []):
        if int(ann.get("category_id", -1)) in BDD_CATEGORY_MAP:
            anns_by_image[int(ann["image_id"])].append(ann)
    result: List[VideoRecord] = []
    missing: List[str] = []
    counts = Counter()
    for source_video_id in sorted(videos):
        source_video = videos[source_video_id]
        source_name = str(source_video["name"])
        video_id = "bdd_%s_%s" % (domain, source_name)
        frame_records = []
        for image in sorted(images_by_video.get(source_video_id, []), key=lambda x: (int(x.get("frame_id", 0)), int(x["id"]))):
            file_name = str(image["file_name"])
            if not (image_root / file_name).is_file():
                missing.append(file_name)
            annotations = []
            for ann in sorted(anns_by_image.get(int(image["id"]), []), key=lambda x: (int(x.get("instance_id", -1)), int(x["id"]))):
                x, y, w, h = [float(v) for v in ann["bbox"]]
                if w <= 0 or h <= 0:
                    continue
                category_id = int(ann["category_id"])
                class_name = BDD_CATEGORY_MAP[category_id]
                annotations.append(AnnotationRecord(
                    track_id=int(ann["instance_id"]),
                    dataset_category_id=category_id,
                    global_semantic_id=GLOBAL_IDS[class_name],
                    bbox_xyxy=[x, y, x + w, y + h],
                    label_source="gt",
                    label_status="reliable" if not int(ann.get("iscrowd", 0)) else "ignore",
                    iscrowd=int(ann.get("iscrowd", 0)),
                ))
                counts[class_name] += 1
            frame_index = int(image.get("frame_id", 0))
            frame_records.append(FrameRecord(
                frame_key="%s/%06d" % (video_id, frame_index),
                frame_index=frame_index,
                file_name=file_name,
                width=int(image["width"]),
                height=int(image["height"]),
                annotations=annotations,
                timestamp_s=float(frame_index) / float(source_video.get("fps", 5) or 5),
                label_scope="complete",
                supervised_global_ids=list(GLOBAL_IDS.values()),
                source_image_id=int(image["id"]),
            ))
        if frame_records:
            result.append(VideoRecord(
                video_id=video_id,
                split=split,
                width=int(source_video["width"]),
                height=int(source_video["height"]),
                frames=frame_records,
                source_video_id=source_video_id,
                source_video_name=source_name,
                dataset="BDD100K MOT",
                image_root="BDD100K_%s" % domain,
            ))
    return result, missing, counts


def convert_bdd(
    source_specs: Sequence[Tuple[str, str, str, str]],
    output_path: str,
    require_images: bool = True,
) -> dict:
    all_videos: List[VideoRecord] = []
    missing: List[str] = []
    counts = Counter()
    source_records = []
    for annotation, image_root, split, domain in source_specs:
        videos, part_missing, part_counts = _convert_file(Path(annotation), Path(image_root), split, domain)
        all_videos.extend(videos)
        missing.extend(["%s:%s" % (domain, value) for value in part_missing])
        counts.update(part_counts)
        source_records.append({
            "annotation_file": Path(annotation).name,
            "image_root": "BDD100K_%s" % domain,
            "split": split,
            "domain": domain,
            "videos": len(videos),
            "frames": sum(len(v.frames) for v in videos),
            "annotations": sum(len(f.annotations) for v in videos for f in v.frames),
            "sha256": __import__("hashlib").sha256(Path(annotation).read_bytes()).hexdigest(),
        })
    if require_images and missing:
        raise FileNotFoundError("%d BDD frame files are missing; first=%s" % (len(missing), missing[0]))
    registry = tao_bdd_registry()
    payload = {
        "schema_version": "cmot.v1",
        "manifest_kind": "canonical_real_video",
        "source": {
            "dataset": "BDD100K MOT",
            "annotation_format": "COCO-like box_track_20",
            "source_category_id_map": {str(k): v for k, v in sorted(BDD_CATEGORY_MAP.items())},
            "semantic_id_map": {k: int(v) for k, v in GLOBAL_IDS.items()},
            "source_records": source_records,
        },
        "id_namespaces": {
            "dataset_category_id": "BDD box_track_20 category ID",
            "global_semantic_id": "zero-based row in the 1203-row LVIS semantic bank",
            "text_row": "same semantic-bank row, recorded in registry",
            "select_column_id": "position inside a stage active_global_ids tuple",
            "track_id": "BDD instance_id; PL IDs are explicitly namespaced by the view builder",
        },
        "registry": registry.as_dict(),
        "stats": {
            "videos": len(all_videos),
            "frames": sum(len(v.frames) for v in all_videos),
            "annotations": sum(len(f.annotations) for v in all_videos for f in v.frames),
            "class_annotations": dict(sorted(counts.items())),
            "split": {
                split: {
                    "videos": sum(1 for v in all_videos if v.split == split),
                    "frames": sum(len(v.frames) for v in all_videos if v.split == split),
                    "annotations": sum(len(f.annotations) for v in all_videos if v.split == split for f in v.frames),
                }
                for split in sorted(set(v.split for v in all_videos))
            },
            "missing_images": len(missing),
        },
        "videos": [v.as_dict() for v in all_videos],
    }
    payload["manifest_hash"] = canonical_json_hash(payload)
    write_json(output_path, payload)
    return payload["stats"]


def _spec(value: str) -> Tuple[str, str, str, str]:
    # annotation,image_root,split,domain
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError("--source must be annotation,image_root,split,domain")
    return tuple(parts)  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="annotation,image_root,split,domain")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()
    print(json.dumps(convert_bdd([_spec(v) for v in args.source], args.output, not args.allow_missing_images), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
