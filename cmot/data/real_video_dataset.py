"""Small, deterministic TAO/BDD real-video adapter for the OVTR core."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    labels: List[int] = []
    boxes: List[List[float]] = []
    obj_ids: List[int] = []
    dataset_category_ids: List[int] = []
    label_weights: List[float] = []
    label_sources: List[int] = []
    source_to_int = {"gt": 0, "gt_replay": 1, "pl": 2}
    for ann in frame.get("annotations", []):
        gid = int(ann["global_semantic_id"])
        if gid not in active or ann.get("label_status") == "ignore":
            continue
        x0, y0, x1, y1 = [float(v) for v in ann["bbox_xyxy"]]
        x0 = min(max(x0, 0.0), float(frame["width"]))
        x1 = min(max(x1, 0.0), float(frame["width"]))
        y0 = min(max(y0, 0.0), float(frame["height"]))
        y1 = min(max(y1, 0.0), float(frame["height"]))
        if x1 <= x0 or y1 <= y0:
            continue
        # cxcywh is normalized in the original image coordinate system.  The
        # adapter preserves aspect ratio, so the same normalized box is valid
        # after the fixed resize.
        labels.append(gid)
        boxes.append([
            ((x0 + x1) * 0.5) / float(frame["width"]),
            ((y0 + y1) * 0.5) / float(frame["height"]),
            (x1 - x0) / float(frame["width"]),
            (y1 - y0) / float(frame["height"]),
        ])
        obj_ids.append(int(ann["track_id"]))
        dataset_category_ids.append(int(ann.get("dataset_category_id", -1)))
        label_weights.append(float(ann.get("score") if ann.get("score") is not None else 1.0))
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
    """Dataset returning the list-of-frames contract used by OVTR.

    It does not hide future annotations in the input: only the selected
    stage-view annotations for each frame are placed into ``gt_instances``.
    The next frame remains available to the criterion solely as a future
    supervision target for the causal motion loss.
    """

    def __init__(
        self,
        view_path: str,
        image_root: str,
        active_global_ids: Sequence[int],
        clip_len: int = 2,
        input_size: Tuple[int, int] = (640, 360),
        samples: int = 1000,
        split: Optional[str] = None,
    ):
        with open(view_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.view_path = str(view_path)
        self.image_root = Path(image_root)
        self.active_global_ids = tuple(int(v) for v in active_global_ids)
        self.clip_len = int(clip_len)
        self.input_size = tuple(int(v) for v in input_size)
        self.samples = int(samples)
        self.split = split or payload.get("split", "train")
        self.videos = [v for v in payload.get("videos", []) if v.get("split") == self.split]
        self.videos = [v for v in self.videos if v.get("frames")]
        if not self.videos:
            raise ValueError("no videos in view %s for split %s" % (view_path, self.split))
        self._video_offsets = []
        running = 0
        for video in self.videos:
            self._video_offsets.append(running)
            running += max(1, len(video["frames"]))
        self._length = running

    def __len__(self) -> int:
        return self.samples

    def _locate(self, index: int) -> Tuple[dict, int]:
        # A deterministic cyclic index keeps long pilot runs independent of
        # Python hash randomization and DataLoader worker scheduling.
        position = int(index) % self._length
        for video, offset in reversed(list(zip(self.videos, self._video_offsets))):
            if position >= offset:
                return video, position - offset
        return self.videos[0], 0

    def __getitem__(self, index: int) -> dict:
        video, start = self._locate(index)
        frames = video["frames"]
        indices = [min(start + j, len(frames) - 1) for j in range(self.clip_len)]
        images: List[torch.Tensor] = []
        targets: List[Instances] = []
        metadata: List[dict] = []
        input_width, input_height = self.input_size
        image_size = (input_height, input_width)
        for frame_index in indices:
            frame = frames[frame_index]
            video_root = video.get("image_root")
            image_path = self.image_root / video_root / frame["file_name"] if video_root else self.image_root / frame["file_name"]
            images.append(_read_image(image_path, self.input_size))
            targets.append(_instances(frame, image_size, self.active_global_ids))
            metadata.append({
                "frame_key": frame["frame_key"],
                "video_id": video["video_id"],
                "frame_index": int(frame["frame_index"]),
                "timestamp_s": frame.get("timestamp_s"),
                "label_scope": frame.get("label_scope", "partial"),
                "supervised_global_ids": [int(v) for v in frame.get("supervised_global_ids", [])],
                "active_global_ids": list(self.active_global_ids),
                "image_size": [int(frame["height"]), int(frame["width"])],
            })
        return {"imgs": images, "gt_instances": targets, "frame_metadata": metadata}


def mot_collate_fn(batch: List[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("C-MOT real-video runner currently requires batch_size=1")
    return batch[0]
