"""Deterministic real-video current/replay clip dataset for OVTR."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from detectron2.structures import Instances


MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)


def _read_image(path: Path, input_size: Tuple[int, int]) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_width, input_height = input_size
    image = cv2.resize(image, (int(input_width), int(input_height)), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    return (tensor - MEAN) / STD


def _instances(frame: dict, image_size: Tuple[int, int], active_global_ids: Sequence[int]) -> Instances:
    height, width = image_size
    active = set(int(v) for v in active_global_ids)
    labels, boxes, obj_ids = [], [], []
    dataset_category_ids, label_weights, label_sources = [], [], []
    source_to_int = {"gt": 0, "gt_replay": 1, "pl": 2}
    for ann in frame.get("annotations", []):
        gid = int(ann["global_semantic_id"])
        if gid not in active or ann.get("label_status") == "ignore" or int(ann.get("iscrowd", 0)):
            continue
        x0, y0, x1, y1 = [float(v) for v in ann["bbox_xyxy"]]
        x0 = min(max(x0, 0.0), float(frame["width"]))
        x1 = min(max(x1, 0.0), float(frame["width"]))
        y0 = min(max(y0, 0.0), float(frame["height"]))
        y1 = min(max(y1, 0.0), float(frame["height"]))
        if x1 <= x0 or y1 <= y0 or not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            continue
        labels.append(gid)
        boxes.append([
            ((x0 + x1) * 0.5) / float(frame["width"]),
            ((y0 + y1) * 0.5) / float(frame["height"]),
            (x1 - x0) / float(frame["width"]),
            (y1 - y0) / float(frame["height"]),
        ])
        obj_ids.append(int(ann["track_id"]))
        dataset_category_ids.append(int(ann.get("dataset_category_id", -1)))
        score = ann.get("score")
        label_weights.append(1.0 if ann.get("label_source") != "pl" or score is None else max(0.0, min(1.0, float(score))))
        label_sources.append(source_to_int.get(ann.get("label_source", "gt"), -1))
    target = Instances(
        image_size,
        labels=torch.as_tensor(labels, dtype=torch.long),
        boxes=torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        obj_ids=torch.as_tensor(obj_ids, dtype=torch.long),
        dataset_category_ids=torch.as_tensor(dataset_category_ids, dtype=torch.long),
        label_weights=torch.as_tensor(label_weights, dtype=torch.float32),
        label_sources=torch.as_tensor(label_sources, dtype=torch.long),
    )
    return target


class ContinualVideoDataset(Dataset):
    """Index actual same-video windows; it never pads a short clip."""

    def __init__(
        self,
        view_path: str,
        image_root: str,
        active_global_ids: Sequence[int],
        clip_len: int = 4,
        input_size: Tuple[int, int] = (640, 360),
        samples: Optional[int] = None,
        split: Optional[str] = None,
        replay_path: Optional[str] = None,
        clip_strides: Sequence[int] = (1,),
        focus_global_ids: Optional[Sequence[int]] = None,
    ):
        self.view_path = str(view_path)
        self.image_root = Path(image_root)
        self.active_global_ids = tuple(int(v) for v in active_global_ids)
        self.focus_global_ids = set(int(v) for v in (focus_global_ids or []))
        self.clip_len = int(clip_len)
        self.input_size = tuple(int(v) for v in input_size)
        self.split = split or "train"
        self.stream_views = []
        for stream, path in (("current", view_path), ("replay", replay_path)):
            if not path:
                continue
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            for video in payload.get("videos", []):
                if video.get("split", self.split) != self.split or not video.get("frames"):
                    continue
                value = dict(video)
                value["stream"] = stream if replay_path else value.get("stream", stream)
                self.stream_views.append(value)
        if not self.stream_views:
            raise ValueError("no videos in repair_v2 dataset")
        self.clip_index = self.build_clip_index(clip_strides)
        if not self.clip_index:
            raise ValueError("no legal clip windows for clip_len=%d" % self.clip_len)

    def build_clip_index(self, clip_strides: Sequence[int] = (1,)) -> List[dict]:
        index = []
        for video in self.stream_views:
            frames = sorted(video["frames"], key=lambda f: (int(f["frame_index"]), f["frame_key"]))
            for stride in sorted(set(int(v) for v in clip_strides)):
                if stride < 1:
                    raise ValueError("clip stride must be positive")
                last_start = len(frames) - (self.clip_len - 1) * stride
                for start in range(max(0, last_start)):
                    selected = [frames[start + j * stride] for j in range(self.clip_len)]
                    if any(int(selected[j + 1]["frame_index"]) <= int(selected[j]["frame_index"]) for j in range(self.clip_len - 1)):
                        continue
                    timestamps = [f.get("timestamp_s") for f in selected]
                    time_valid = all(t is not None and math.isfinite(float(t)) for t in timestamps) and all(
                        float(timestamps[j + 1]) > float(timestamps[j]) for j in range(self.clip_len - 1)
                    )
                    focus = sorted({
                        int(a["global_semantic_id"])
                        for f in selected
                        for a in f.get("annotations", [])
                        if int(a.get("global_semantic_id", -1)) in self.focus_global_ids
                    })
                    index.append({
                        "stream": video.get("stream", "current"),
                        "video_id": str(video["video_id"]),
                        "source_video_id": str(video.get("source_video_id", video["video_id"])),
                        "image_root": video.get("image_root"),
                        "frames": selected,
                        "start": int(start),
                        "stride": int(stride),
                        "focus_global_ids": focus,
                        "focus_class": focus[0] if focus else None,
                        "motion_time_valid": bool(time_valid),
                        "clip_id": "%s:%s:%d:%d" % (video["video_id"], video.get("stream", "current"), start, stride),
                    })
        return index

    def __len__(self) -> int:
        return len(self.clip_index)

    def __getitem__(self, index: int) -> dict:
        item = self.clip_index[int(index)]
        images, targets, metadata = [], [], []
        input_width, input_height = self.input_size
        image_size = (input_height, input_width)
        for frame in item["frames"]:
            video_root = item.get("image_root")
            image_path = self.image_root / video_root / frame["file_name"] if video_root else self.image_root / frame["file_name"]
            images.append(_read_image(image_path, self.input_size))
            targets.append(_instances(frame, image_size, self.active_global_ids))
            timestamp = frame.get("timestamp_s")
            metadata.append({
                "frame_key": frame["frame_key"],
                "video_id": item["video_id"],
                "source_video_id": item["source_video_id"],
                "frame_index": int(frame["frame_index"]),
                "timestamp_s": timestamp,
                "motion_time_valid": bool(item["motion_time_valid"]),
                "label_scope": frame.get("label_scope", "partial"),
                "supervised_global_ids": [int(v) for v in frame.get("exhaustive_global_ids", frame.get("supervised_global_ids", []))],
                "exhaustive_global_ids": [int(v) for v in frame.get("exhaustive_global_ids", frame.get("supervised_global_ids", []))],
                "ignore_regions": list(frame.get("ignore_regions", [])),
                "annotation_valid": bool(frame.get("annotation_valid", True)),
                "active_global_ids": list(self.active_global_ids),
                "image_size": [int(frame["height"]), int(frame["width"])],
                "supervision_protocol": frame.get("supervision_protocol", frame.get("label_scope", "partial")),
            })
        return {
            "imgs": images,
            "gt_instances": targets,
            "frame_metadata": metadata,
            "sample_metadata": {
                "clip_id": item["clip_id"],
                "stream": item.get("stream", "current"),
                "focus_class": item.get("focus_class"),
                "focus_global_ids": item.get("focus_global_ids", []),
                "source_video_uid": item["source_video_id"],
                "augmentation_seed": int(hashlib.sha256(item["clip_id"].encode("utf-8")).hexdigest()[:8], 16),
            },
        }


def mot_collate_fn(batch: List[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("C-MOT real-video runner currently requires batch_size=1")
    return batch[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--active-global-ids", required=True)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=360)
    args = parser.parse_args()
    dataset = ContinualVideoDataset(
        args.view, args.image_root, [int(v) for v in args.active_global_ids.split(",") if v],
        clip_len=args.clip_len, input_size=(args.input_width, args.input_height),
    )
    print(json.dumps({"status": "OK", "clips": len(dataset), "videos": len(dataset.stream_views)}, sort_keys=True))


if __name__ == "__main__":
    main()
