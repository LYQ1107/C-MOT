"""COOLer-compatible BDD100K M→L views and two-frame sampling.

This module is deliberately independent from the V3/V4 view builders.  A
COOLer stage never reads an earlier stage's real training annotation.  The
only old-class rows that can be merged into an S1/S2 view are records emitted
by the previous-stage tracker on the current stage's videos.
"""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..manifest import canonical_json_hash, sha256_file, write_json


CLASS_IDS = {"car": 206, "pedestrian": 792, "truck": 1122}
CLASS_NAMES = {value: key for key, value in CLASS_IDS.items()}
BDD_CATEGORY_NAMES = {1: "pedestrian", 2: "car", 3: "truck"}


def _read_payload(source: Any) -> Tuple[dict, Optional[str], str]:
    if isinstance(source, Mapping):
        return deepcopy(dict(source)), None, "mapping"
    path = Path(str(source))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("BDD source manifest must be a JSON object")
    return payload, str(path), path.name


def _source_hash(payload: Mapping[str, Any], path: Optional[str]) -> str:
    if path:
        return sha256_file(path)
    return str(payload.get("manifest_hash") or canonical_json_hash(payload))


def _global_id(annotation: Mapping[str, Any], categories: Optional[Mapping[int, str]] = None) -> Optional[int]:
    if annotation.get("global_semantic_id") is not None:
        value = int(annotation["global_semantic_id"])
        return value if value in CLASS_NAMES else None
    category_id = int(annotation.get("category_id", -1))
    name = (categories or {}).get(category_id) or BDD_CATEGORY_NAMES.get(category_id)
    return CLASS_IDS.get(str(name).strip().lower()) if name else None


def _coco_to_videos(payload: Mapping[str, Any], default_split: str) -> List[dict]:
    """Normalize a COCO-like box_track_20 object without changing the source."""
    videos = {int(item["id"]): dict(item) for item in payload.get("videos", []) if "id" in item}
    categories = {
        int(item["id"]): str(item.get("name", "")).strip().lower()
        for item in payload.get("categories", [])
        if "id" in item
    }
    images_by_video: Dict[int, List[dict]] = defaultdict(list)
    for image in payload.get("images", []):
        if "video_id" in image:
            images_by_video[int(image["video_id"])].append(dict(image))
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if "image_id" in ann:
            anns_by_image[int(ann["image_id"])].append(dict(ann))
    result = []
    for video_id, raw_video in sorted(videos.items()):
        split = str(raw_video.get("split") or default_split)
        source_name = str(raw_video.get("name") or raw_video.get("video_name") or video_id)
        source_uid = str(raw_video.get("source_video_uid") or source_name)
        frames = []
        for image in sorted(images_by_video.get(video_id, []), key=lambda value: (int(value.get("frame_id", value.get("frame_index", 0))), int(value.get("id", 0)))):
            frame_index = int(image.get("frame_id", image.get("frame_index", 0)))
            annotations = []
            for ann in sorted(anns_by_image.get(int(image.get("id", -1)), []), key=lambda value: int(value.get("instance_id", value.get("track_id", value.get("id", -1))))):
                gid = _global_id(ann, categories)
                if gid is None:
                    # BDD distractors and classes outside this benchmark are
                    # not C-MOT targets; the exact evaluator handles them as
                    # ignore rows when they are present in the source view.
                    continue
                bbox = ann.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                x, y, width, height = [float(value) for value in bbox]
                if width <= 0 or height <= 0:
                    continue
                track_id = ann.get("instance_id", ann.get("track_id", ann.get("id")))
                if track_id is None:
                    continue
                annotations.append({
                    "track_id": int(track_id),
                    "dataset_category_id": int(ann.get("category_id", -1)),
                    "global_semantic_id": gid,
                    "bbox_xyxy": [x, y, x + width, y + height],
                    "label_source": "gt",
                    "label_status": "ignore" if int(ann.get("iscrowd", 0)) else "reliable",
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "track_uid": "%s:track:%s" % (source_uid, track_id),
                })
            frames.append({
                "frame_key": "%s/%06d" % (source_name, frame_index),
                "frame_uid": "%s:frame:%06d" % (source_uid, frame_index),
                "source_video_uid": source_uid,
                "frame_index": frame_index,
                "file_name": str(image["file_name"]),
                "width": int(image.get("width", raw_video.get("width", 1280))),
                "height": int(image.get("height", raw_video.get("height", 720))),
                "timestamp_s": float(frame_index) / float(raw_video.get("fps", 5) or 5),
                "annotations": annotations,
            })
        if frames:
            result.append({
                "video_id": "bdd_%s" % source_name,
                "source_video_id": video_id,
                "source_video_name": source_name,
                "source_video_uid": source_uid,
                "split": split,
                "dataset": "BDD100K MOT",
                "image_root": raw_video.get("image_root"),
                "width": int(raw_video.get("width", 1280)),
                "height": int(raw_video.get("height", 720)),
                "frames": frames,
            })
    return result


def _videos_from_payload(payload: Mapping[str, Any], default_split: str) -> List[dict]:
    if "images" in payload and "annotations" in payload:
        return _coco_to_videos(payload, default_split)
    result = []
    for raw_video in payload.get("videos", []):
        video = deepcopy(raw_video)
        video.setdefault("split", default_split)
        video.setdefault("source_video_uid", video.get("source_video_id", video.get("video_id")))
        for frame in video.get("frames", []):
            frame.setdefault("source_video_uid", video["source_video_uid"])
            frame.setdefault("frame_uid", "%s:frame:%06d" % (video["source_video_uid"], int(frame.get("frame_index", 0))))
            for ann in frame.get("annotations", []):
                ann.setdefault("track_uid", "%s:track:%s" % (video["source_video_uid"], ann.get("track_id")))
        result.append(video)
    return result


def _annotation_copy(annotation: Mapping[str, Any], source: str, acquired_stage: str) -> dict:
    value = deepcopy(dict(annotation))
    value["label_source"] = source
    value["label_status"] = "reliable" if source == "gt" else "pseudo"
    value["acquired_stage"] = acquired_stage
    if source == "gt":
        value["teacher_checkpoint_sha256"] = None
        value["pl_segment_id"] = None
        value["reliability"] = 1.0
    return value


def _video_source_uid(video: Mapping[str, Any]) -> str:
    return str(video.get("source_video_uid") or video.get("source_video_id") or video.get("video_id"))


def build_cooler_stage_view(
    raw_bdd_manifest: Any,
    stage_id: str,
    new_class_names: Sequence[str],
    seen_class_names: Sequence[str],
    split: str,
) -> dict:
    """Build a train or validation view using COOLer's stage access rule.

    Train views contain only current-new GT.  Non-train views contain the
    declared seen GT for evaluation.  The function returns a new object and
    never edits a source manifest loaded from disk.
    """
    payload, source_path, source_basename = _read_payload(raw_bdd_manifest)
    new_names = [str(name).strip().lower() for name in new_class_names]
    seen_names = [str(name).strip().lower() for name in seen_class_names]
    if not new_names or not set(new_names).issubset(set(seen_names)):
        raise ValueError("new classes must be a non-empty subset of seen classes")
    new_ids = [CLASS_IDS[name] for name in new_names]
    seen_ids = [CLASS_IDS[name] for name in seen_names]
    videos = _videos_from_payload(payload, split)
    output_videos = []
    stats = Counter()
    for raw_video in videos:
        if str(raw_video.get("split")) != str(split):
            continue
        source_uid = _video_source_uid(raw_video)
        raw_frames = sorted(raw_video.get("frames", []), key=lambda frame: (int(frame.get("frame_index", 0)), str(frame.get("frame_key", ""))))
        has_new = any(
            int(ann.get("global_semantic_id", -1)) in new_ids
            and ann.get("label_source", "gt") == "gt"
            and not int(ann.get("iscrowd", 0))
            for frame in raw_frames for ann in frame.get("annotations", [])
        )
        if split == "train" and not has_new:
            continue
        frames = []
        for raw_frame in raw_frames:
            frame = deepcopy(raw_frame)
            if split == "train":
                allowed_ids = set(new_ids)
                source_label = "cooler_current_new_gt"
                label_scope = "cooler_complete_seen"
            else:
                allowed_ids = set(seen_ids)
                source_label = "immutable_eval_gt"
                label_scope = "immutable_eval_gt"
            kept = []
            for raw_ann in raw_frame.get("annotations", []):
                gid = int(raw_ann.get("global_semantic_id", -1))
                if gid not in allowed_ids or raw_ann.get("label_source", "gt") != "gt":
                    continue
                kept.append(_annotation_copy(raw_ann, "gt", source_label))
                stats["gt_annotations_by_class:%s" % CLASS_NAMES[gid]] += 1
            frame["source_video_uid"] = source_uid
            frame["annotations"] = kept
            frame["label_scope"] = label_scope
            frame["supervised_global_ids"] = list(seen_ids)
            frame["exhaustive_global_ids"] = list(seen_ids)
            frame["annotation_valid"] = True
            frames.append(frame)
        if frames:
            video = deepcopy(raw_video)
            video["source_video_uid"] = source_uid
            video["frames"] = frames
            video["stage"] = str(stage_id)
            video["protocol_role"] = "cooler_compat"
            video["label_mode"] = "cooler_complete_seen" if split == "train" else "immutable_eval_gt"
            output_videos.append(video)
            stats["source_videos"] += 1
            stats["frames"] += len(frames)
    for name in new_names:
        stats["gt_annotations_by_class:%s" % name] += 0
    result = {
        "schema_version": "cmot.cooler_compat.view.v1",
        "stage": str(stage_id),
        "split": str(split),
        "active_global_ids": list(seen_ids),
        "new_global_ids": list(new_ids),
        "old_global_ids": [value for value in seen_ids if value not in new_ids],
        "label_mode": "cooler_complete_seen" if split == "train" else "immutable_eval_gt",
        "source_manifest_basename": source_basename,
        "source_manifest_sha256": _source_hash(payload, source_path),
        "videos": output_videos,
        "stats": {
            "source_videos": int(stats["source_videos"]),
            "frames": int(stats["frames"]),
            "gt_annotations_by_class": {
                name: int(stats["gt_annotations_by_class:%s" % name]) for name in seen_names
            },
            "hidden_old_gt_annotations_loaded": 0,
            "eligible_videos_have_new_gt": True,
        },
        "leakage_audit": {
            "stage": str(stage_id),
            "source_videos": int(stats["source_videos"]),
            "frames": int(stats["frames"]),
            "gt_annotations_by_class": {
                name: int(stats["gt_annotations_by_class:%s" % name]) for name in seen_names
            },
            "hidden_old_gt_annotations_loaded": 0,
        },
    }
    result["manifest_hash"] = canonical_json_hash(result)
    return result


def view_counts(view: Mapping[str, Any]) -> dict:
    counts = Counter()
    tracks = defaultdict(set)
    for video in view.get("videos", []):
        uid = _video_source_uid(video)
        counts["videos"] += 1
        for frame in video.get("frames", []):
            counts["frames"] += 1
            for ann in frame.get("annotations", []):
                source = str(ann.get("label_source", "gt"))
                gid = int(ann.get("global_semantic_id", -1))
                if source == "gt":
                    counts["gt_boxes"] += 1
                    counts["gt_boxes:%s" % CLASS_NAMES.get(gid, str(gid))] += 1
                elif source == "pl":
                    counts["pseudo_boxes"] += 1
                    counts["pseudo_boxes:%s" % CLASS_NAMES.get(gid, str(gid))] += 1
                if source in ("gt", "pl"):
                    tracks[(source, gid)].add((uid, str(ann.get("track_uid") or ann.get("track_id"))))
    counts["gt_tracks"] = len({item for key, values in tracks.items() if key[0] == "gt" for item in values})
    counts["pseudo_tracks"] = len({item for key, values in tracks.items() if key[0] == "pl" for item in values})
    return dict(counts)


def merge_cooler_track_pl(
    stage_view: Mapping[str, Any],
    pseudo_jsonl: str,
    output_path: Optional[str] = None,
    old_global_ids: Sequence[int] = (),
) -> dict:
    """Merge previous-tracker records into current videos only."""
    result = deepcopy(dict(stage_view))
    allowed = {int(value) for value in old_global_ids}
    pseudo_by_frame: Dict[str, List[dict]] = defaultdict(list)
    metadata = {}
    with Path(pseudo_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("record_type") == "metadata":
                metadata.update(value.get("metadata", value))
                continue
            for prediction in value.get("predictions", []):
                gid = int(prediction.get("global_id", -1))
                if gid not in allowed:
                    continue
                row = {
                    "track_id": int(prediction["track_id"]),
                    "dataset_category_id": -1,
                    "global_semantic_id": gid,
                    "bbox_xyxy": [float(v) for v in prediction["bbox_xyxy"]],
                    "label_source": "pl",
                    "label_status": "pseudo",
                    "score": float(prediction.get("score", 0.0)),
                    "iscrowd": 0,
                    "acquired_stage": "cooler_track_pl_v1",
                    "track_uid": "%s:pl_track:%s" % (value.get("source_video_uid", value.get("video_id", "unknown")), prediction["track_id"]),
                    "teacher_checkpoint_sha256": prediction.get(
                        "teacher_checkpoint_sha256", metadata.get("teacher_checkpoint_sha256")
                    ),
                    "reliability": float(prediction.get("score", 0.0)),
                }
                pseudo_by_frame[str(value.get("frame_key"))].append(row)
    required_metadata = (
        "policy_version", "teacher_checkpoint_sha256", "current_stage_data_sha256",
        "allowed_pseudo_global_ids", "new_gt_exclusion_iou",
    )
    missing_metadata = [key for key in required_metadata if metadata.get(key) in (None, "")]
    if missing_metadata:
        raise ValueError("COOLer pseudo manifest metadata missing: %s" % ",".join(missing_metadata))
    if metadata.get("policy_version") != "cooler_track_pl_v1":
        raise ValueError("unexpected COOLer pseudo policy: %s" % metadata.get("policy_version"))
    if {int(value) for value in metadata.get("allowed_pseudo_global_ids", [])} != allowed:
        raise ValueError("COOLer pseudo old-class binding mismatch")
    if abs(float(metadata.get("new_gt_exclusion_iou")) - 0.5) > 1e-9:
        raise ValueError("COOLer pseudo exclusion IoU must be 0.5")
    added = Counter()
    overlap = 0
    for video in result.get("videos", []):
        for frame in video.get("frames", []):
            rows = pseudo_by_frame.get(str(frame.get("frame_key")), [])
            if not rows:
                continue
            existing = {
                (int(ann.get("global_semantic_id", -1)), tuple(float(v) for v in ann.get("bbox_xyxy", [])))
                for ann in frame.get("annotations", [])
            }
            for row in rows:
                key = (int(row["global_semantic_id"]), tuple(row["bbox_xyxy"]))
                if key in existing:
                    overlap += 1
                    continue
                frame.setdefault("annotations", []).append(row)
                added[CLASS_NAMES.get(int(row["global_semantic_id"]), str(row["global_semantic_id"]))] += 1
    result["pseudo_metadata"] = dict(metadata)
    result["pseudo_merge_audit"] = {
        "policy_version": "cooler_track_pl_v1",
        "allowed_old_global_ids": sorted(allowed),
        "pseudo_boxes_added": int(sum(added.values())),
        "pseudo_boxes_by_class": dict(sorted(added.items())),
        "duplicate_rows_ignored": int(overlap),
        "gt_replay": 0,
    }
    result.pop("manifest_hash", None)
    result["manifest_hash"] = canonical_json_hash(result)
    if output_path:
        write_json(output_path, result)
    return result


def _track_keys(frame: Mapping[str, Any]) -> set:
    result = set()
    for ann in frame.get("annotations", []):
        if ann.get("label_source", "gt") not in ("gt", "pl"):
            continue
        if ann.get("label_status") == "ignore" or int(ann.get("iscrowd", 0)):
            continue
        result.add(str(ann.get("track_uid") or ann.get("track_id")))
    return result


class CoolerCompatiblePairDataset:
    """Nominal-frame dataset implementing COOLer's key+uniform-reference rule."""

    def __init__(
        self,
        view: Any,
        image_root: str,
        active_global_ids: Sequence[int],
        reference_scope: int = 3,
        horizontal_flip_probability: float = 0.5,
    ):
        if isinstance(view, Mapping):
            payload = deepcopy(dict(view))
        else:
            payload = json.loads(Path(str(view)).read_text(encoding="utf-8"))
        self.payload = payload
        self.view_path = None if isinstance(view, Mapping) else str(view)
        self.image_root = Path(image_root)
        self.active_global_ids = tuple(int(value) for value in active_global_ids)
        self.reference_scope = int(reference_scope)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        if self.reference_scope < 1:
            raise ValueError("reference_scope must be positive")
        self.videos = []
        self.nominal = []
        self.frames_by_video = []
        self.valid_key_indices = []
        for video_index, raw_video in enumerate(sorted(payload.get("videos", []), key=lambda value: str(value.get("video_id")))):
            if str(raw_video.get("split", "train")) != "train":
                continue
            video = deepcopy(raw_video)
            video["frames"] = sorted(video.get("frames", []), key=lambda value: (int(value.get("frame_index", 0)), str(value.get("frame_key", ""))))
            if not video["frames"]:
                continue
            local_video_index = len(self.videos)
            self.videos.append(video)
            self.frames_by_video.append(video["frames"])
            for frame_index in range(len(video["frames"])):
                nominal_index = len(self.nominal)
                self.nominal.append((local_video_index, frame_index))
                if self._reference_candidates(local_video_index, frame_index):
                    self.valid_key_indices.append(nominal_index)
        if not self.nominal:
            raise ValueError("COOLer stage view has no training frames")
        if not self.valid_key_indices:
            raise ValueError("COOLer stage view has no legal key/reference pair")
        self.epoch = 0
        self.epoch_order = list(range(len(self.nominal)))
        self.fallback_count = 0
        self.invalid_nominal_count = len(self.nominal) - len(self.valid_key_indices)

    def __len__(self) -> int:
        return len(self.nominal)

    @property
    def nominal_length(self) -> int:
        return len(self.nominal)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.epoch_order = list(range(len(self.nominal)))
        rng = random.Random(777 + self.epoch)
        rng.shuffle(self.epoch_order)
        self.fallback_count = 0

    def _hash_int(self, *values: Any) -> int:
        value = "|".join(str(item) for item in values).encode("utf-8")
        return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")

    def _reference_candidates(self, video_index: int, frame_index: int) -> List[int]:
        frames = self.frames_by_video[video_index]
        key = frames[frame_index]
        key_tracks = _track_keys(key)
        if not key_tracks:
            return []
        key_number = int(key.get("frame_index", frame_index))
        candidates = []
        for candidate_index, candidate in enumerate(frames):
            if candidate_index == frame_index:
                continue
            candidate_number = int(candidate.get("frame_index", candidate_index))
            if abs(candidate_number - key_number) > self.reference_scope:
                continue
            if key_tracks.intersection(_track_keys(candidate)):
                candidates.append(candidate_index)
        return sorted(candidates, key=lambda value: (abs(int(frames[value].get("frame_index", value)) - key_number), int(frames[value].get("frame_index", value)), value))

    def _select_pair(self, nominal_index: int) -> Tuple[int, int, int, bool, List[str]]:
        video_index, key_index = self.nominal[int(nominal_index)]
        candidates = self._reference_candidates(video_index, key_index)
        fallback = False
        if not candidates:
            fallback = True
            # Deterministically retry another nominal key from the same
            # stage.  This preserves nominal epoch length without ever
            # returning a key/reference pair with no shared identity.
            start = self._hash_int(777 + self.epoch, nominal_index, "fallback") % len(self.valid_key_indices)
            for offset in range(len(self.valid_key_indices)):
                candidate_nominal = self.valid_key_indices[(start + offset) % len(self.valid_key_indices)]
                candidate_video, candidate_key = self.nominal[candidate_nominal]
                candidate_refs = self._reference_candidates(candidate_video, candidate_key)
                if candidate_refs:
                    video_index, key_index, candidates = candidate_video, candidate_key, candidate_refs
                    break
            else:
                raise RuntimeError("deterministic COOLer fallback found no legal pair")
        ref_offset = self._hash_int(777 + self.epoch, nominal_index, video_index, key_index, "reference") % len(candidates)
        reference_index = candidates[ref_offset]
        frames = self.frames_by_video[video_index]
        common = sorted(_track_keys(frames[key_index]).intersection(_track_keys(frames[reference_index])))
        if not common:
            raise RuntimeError("COOLer pair selected without a shared identity")
        if fallback:
            self.fallback_count += 1
        return video_index, key_index, reference_index, fallback, common

    def _flip_for(self, nominal_index: int) -> bool:
        threshold = int(max(0.0, min(1.0, self.horizontal_flip_probability)) * ((1 << 64) - 1))
        return self._hash_int(777 + self.epoch, nominal_index, "flip") <= threshold

    @staticmethod
    def _flip_target(target) -> None:
        if len(target):
            target.boxes[:, 0] = 1.0 - target.boxes[:, 0]

    def __getitem__(self, index: int) -> dict:
        import torch

        from .real_video_dataset import _instances, _read_image

        nominal_index = int(self.epoch_order[int(index)])
        video_index, key_index, reference_index, fallback, common = self._select_pair(nominal_index)
        video = self.videos[video_index]
        frames = self.frames_by_video[video_index]
        selected_indices = sorted((key_index, reference_index), key=lambda value: (int(frames[value].get("frame_index", value)), str(frames[value].get("frame_key", ""))))
        flip = self._flip_for(nominal_index)
        images, targets, metadata = [], [], []
        for frame_index in selected_indices:
            frame = frames[frame_index]
            root = video.get("image_root")
            image_path = self.image_root / root / frame["file_name"] if root else self.image_root / frame["file_name"]
            image = _read_image(image_path, (1280, 720))
            target = _instances(frame, (720, 1280), self.active_global_ids)
            if flip:
                image = torch.flip(image, dims=[2])
                self._flip_target(target)
            images.append(image)
            metadata.append({
                "frame_key": frame["frame_key"],
                "frame_uid": str(frame.get("frame_uid") or frame["frame_key"]),
                "video_id": video["video_id"],
                "source_video_uid": _video_source_uid(video),
                "frame_index": int(frame["frame_index"]),
                "timestamp_s": frame.get("timestamp_s"),
                "image_size": [int(frame["height"]), int(frame["width"])],
                "active_global_ids": list(self.active_global_ids),
                "supervised_global_ids": [int(value) for value in frame.get("supervised_global_ids", self.active_global_ids)],
                "exhaustive_global_ids": [int(value) for value in frame.get("exhaustive_global_ids", self.active_global_ids)],
                "label_scope": frame.get("label_scope", "cooler_complete_seen"),
                "supervision_protocol": frame.get("label_mode", "cooler_complete_seen"),
                "annotation_valid": bool(frame.get("annotation_valid", True)),
                "ignore_regions": list(frame.get("ignore_regions", [])),
                "horizontal_flip": bool(flip),
                "pair_consistent_flip": True,
            })
        return {
            "imgs": images,
            "gt_instances": targets,
            "frame_metadata": metadata,
            "sample_metadata": {
                "clip_id": "%s:nominal:%d:epoch:%d" % (video["video_id"], nominal_index, self.epoch),
                "stream": "current",
                "source_video_uid": _video_source_uid(video),
                "video_id": video["video_id"],
                "nominal_index": nominal_index,
                "epoch": int(self.epoch),
                "key_frame_index": int(frames[key_index]["frame_index"]),
                "reference_frame_index": int(frames[reference_index]["frame_index"]),
                "common_track_uids": common,
                "fallback_used": bool(fallback),
                "horizontal_flip": bool(flip),
                "pair_consistent_flip": True,
                "has_pl": any(int(source) == 2 for target in targets for source in target.label_sources.tolist()) if targets else False,
            },
        }

    def sampling_plan(self) -> List[dict]:
        original_fallback = self.fallback_count
        self.fallback_count = 0
        plan = []
        for loader_index, nominal_index in enumerate(self.epoch_order):
            video_index, key_index, reference_index, fallback, common = self._select_pair(nominal_index)
            plan.append({
                "loader_index": loader_index,
                "nominal_index": nominal_index,
                "video_id": self.videos[video_index]["video_id"],
                "key_frame_index": int(self.frames_by_video[video_index][key_index]["frame_index"]),
                "reference_frame_index": int(self.frames_by_video[video_index][reference_index]["frame_index"]),
                "common_track_uids": common,
                "fallback_used": bool(fallback),
                "horizontal_flip": self._flip_for(nominal_index),
            })
        self.fallback_count = original_fallback
        return plan

    def plan_hash(self) -> str:
        return canonical_json_hash({"seed": 777 + self.epoch, "epoch": self.epoch, "plan": self.sampling_plan()})


__all__ = [
    "CLASS_IDS", "CLASS_NAMES", "build_cooler_stage_view", "merge_cooler_track_pl",
    "view_counts", "CoolerCompatiblePairDataset",
]
