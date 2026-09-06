# ------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
from .boxes import Boxes, BoxMode, pairwise_iou, pairwise_ioa
from .instances import Instances

__all__ = [k for k in globals().keys() if not k.startswith("_")]

def matched_boxlist_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    
    assert len(boxes1) == len(boxes2), "boxlists should have the same" "number of entries, got {}, {}".format(len(boxes1), len(boxes2))
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou