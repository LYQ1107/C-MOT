"""Canonical C-MOT data and experiment contracts.

The source datasets use different identifiers for categories, semantic rows,
model columns and tracks.  These dataclasses make those namespaces explicit
at the boundary instead of relying on positional conventions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


ID_NAMESPACES = (
    "dataset_category_id",
    "global_semantic_id",
    "select_column_id",
    "track_id",
)


@dataclass(frozen=True)
class SemanticClass:
    name: str
    global_semantic_id: int
    text_row: int
    aliases: Tuple[str, ...] = ()
    exclusive_group: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "global_semantic_id": self.global_semantic_id,
            "text_row": self.text_row,
            "aliases": list(self.aliases),
            "exclusive_group": self.exclusive_group,
        }


@dataclass
class AnnotationRecord:
    track_id: int
    dataset_category_id: int
    global_semantic_id: int
    bbox_xyxy: List[float]
    label_source: str = "gt"
    label_status: str = "reliable"
    score: Optional[float] = None
    iscrowd: int = 0
    acquired_stage: str = ""
    track_uid: str = ""
    teacher_checkpoint_sha256: Optional[str] = None
    pl_segment_id: Optional[str] = None
    reliability: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "dataset_category_id": int(self.dataset_category_id),
            "global_semantic_id": int(self.global_semantic_id),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "label_source": self.label_source,
            "label_status": self.label_status,
            "score": None if self.score is None else float(self.score),
            "iscrowd": int(self.iscrowd),
            "acquired_stage": self.acquired_stage,
            "track_uid": self.track_uid,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "pl_segment_id": self.pl_segment_id,
            "reliability": None if self.reliability is None else float(self.reliability),
        }


@dataclass
class FrameRecord:
    frame_key: str
    frame_index: int
    file_name: str
    width: int
    height: int
    annotations: List[AnnotationRecord] = field(default_factory=list)
    timestamp_s: Optional[float] = None
    label_scope: str = "complete"
    supervised_global_ids: List[int] = field(default_factory=list)
    exhaustive_global_ids: List[int] = field(default_factory=list)
    ignore_regions: List[dict] = field(default_factory=list)
    annotation_valid: bool = True
    source_image_id: Optional[int] = None
    source_video_uid: str = ""
    frame_uid: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame_key": self.frame_key,
            "frame_index": int(self.frame_index),
            "file_name": self.file_name,
            "width": int(self.width),
            "height": int(self.height),
            "timestamp_s": self.timestamp_s,
            "label_scope": self.label_scope,
            # ``supervised_global_ids`` is retained for v1 readers, but v2
            # gives it the same meaning as exhaustiveness.  A PL presence is
            # deliberately not allowed to widen either list.
            "supervised_global_ids": [int(v) for v in self.exhaustive_global_ids],
            "exhaustive_global_ids": [int(v) for v in self.exhaustive_global_ids],
            "ignore_regions": list(self.ignore_regions),
            "annotation_valid": bool(self.annotation_valid),
            "source_image_id": self.source_image_id,
            "source_video_uid": self.source_video_uid,
            "frame_uid": self.frame_uid or self.frame_key,
            "annotations": [a.as_dict() for a in self.annotations],
        }


@dataclass
class VideoRecord:
    video_id: str
    split: str
    width: int
    height: int
    frames: List[FrameRecord]
    source_video_id: Optional[int] = None
    source_video_name: Optional[str] = None
    dataset: str = "TAO-Amodal"
    image_root: Optional[str] = None
    source_video_uid: str = ""

    def as_dict(self) -> Dict[str, Any]:
        # source_video_name is useful only for local resolution.  It is kept in
        # the private canonical manifest but can be removed by the public
        # manifest sanitizer.
        return {
            "video_id": self.video_id,
            "split": self.split,
            "width": int(self.width),
            "height": int(self.height),
            "source_video_id": self.source_video_id,
            "source_video_name": self.source_video_name,
            "dataset": self.dataset,
            "image_root": self.image_root,
            "source_video_uid": self.source_video_uid or self.video_id,
            "frames": [f.as_dict() for f in self.frames],
        }


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    name: str
    active_global_ids: Tuple[int, ...]
    new_global_ids: Tuple[int, ...]
    old_global_ids: Tuple[int, ...]
    train_split: str = "train"
    eval_split: str = "val"
    label_protocol: str = "partial"
    motion_mode: str = "none"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "active_global_ids": list(self.active_global_ids),
            "new_global_ids": list(self.new_global_ids),
            "old_global_ids": list(self.old_global_ids),
            "train_split": self.train_split,
            "eval_split": self.eval_split,
            "label_protocol": self.label_protocol,
            "motion_mode": self.motion_mode,
        }


def annotation_from_dict(value: Dict[str, Any]) -> AnnotationRecord:
    return AnnotationRecord(
        track_id=int(value["track_id"]),
        dataset_category_id=int(value["dataset_category_id"]),
        global_semantic_id=int(value["global_semantic_id"]),
        bbox_xyxy=[float(v) for v in value["bbox_xyxy"]],
        label_source=value.get("label_source", "gt"),
        label_status=value.get("label_status", "reliable"),
        score=value.get("score"),
        iscrowd=int(value.get("iscrowd", 0)),
        acquired_stage=value.get("acquired_stage", ""),
        track_uid=value.get("track_uid", ""),
        teacher_checkpoint_sha256=value.get("teacher_checkpoint_sha256"),
        pl_segment_id=value.get("pl_segment_id"),
        reliability=value.get("reliability"),
    )


def frame_from_dict(value: Dict[str, Any]) -> FrameRecord:
    exhaustive = value.get("exhaustive_global_ids")
    if exhaustive is None:
        # Canonical v1 manifests are accepted as read-only source material.
        # Stage views are rebuilt as cmot.v2 before they can be used for a
        # repair_v2 run.
        exhaustive = value.get("supervised_global_ids", [])
    return FrameRecord(
        frame_key=value["frame_key"],
        frame_index=int(value["frame_index"]),
        file_name=value["file_name"],
        width=int(value["width"]),
        height=int(value["height"]),
        annotations=[annotation_from_dict(a) for a in value.get("annotations", [])],
        timestamp_s=value.get("timestamp_s"),
        label_scope=value.get("label_scope", "complete"),
        supervised_global_ids=[int(v) for v in exhaustive],
        exhaustive_global_ids=[int(v) for v in exhaustive],
        ignore_regions=list(value.get("ignore_regions", [])),
        annotation_valid=bool(value.get("annotation_valid", True)),
        source_image_id=value.get("source_image_id"),
        source_video_uid=value.get("source_video_uid", ""),
        frame_uid=value.get("frame_uid", ""),
    )


def video_from_dict(value: Dict[str, Any]) -> VideoRecord:
    source_video_uid = str(value.get("source_video_uid") or value.get("video_id"))
    frames = [frame_from_dict(f) for f in value.get("frames", [])]
    # Older manifests did not carry stable source/frame UIDs.  Derive them at
    # the read boundary without changing the source file on disk.
    for frame in frames:
        if not frame.source_video_uid:
            frame.source_video_uid = source_video_uid
        if not frame.frame_uid:
            frame.frame_uid = "%s:%06d" % (source_video_uid, frame.frame_index)
        for ann in frame.annotations:
            if not ann.track_uid:
                ann.track_uid = "%s:%s" % (source_video_uid, ann.track_id)
    return VideoRecord(
        video_id=value["video_id"],
        split=value["split"],
        width=int(value["width"]),
        height=int(value["height"]),
        frames=frames,
        source_video_id=value.get("source_video_id"),
        source_video_name=value.get("source_video_name"),
        dataset=value.get("dataset", "TAO-Amodal"),
        image_root=value.get("image_root"),
        source_video_uid=source_video_uid,
    )
