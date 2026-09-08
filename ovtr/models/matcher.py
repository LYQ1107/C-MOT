# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from detectron2.structures import Instances


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, use_focal=True):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # Also concat the target labels and boxes
            if isinstance(targets[0], Instances):
                tgt_ids = torch.cat([gt_per_img.labels for gt_per_img in targets])
                tgt_bbox = torch.cat([gt_per_img.boxes for gt_per_img in targets])
            else:
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])

            select_id = outputs["select_id"]
            tgt_ids_all = [(select_id == lid).nonzero(as_tuple=False)[0] for lid in tgt_ids]

            # We flatten to compute the cost matrices in a batch
            if use_focal:
                out_prob = outputs["pred_logits"][:, :, tgt_ids_all].flatten(0, 1).sigmoid()
            else:
                out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes_for_matching_pre"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Compute the classification cost.
            if use_focal:
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            if isinstance(targets[0], Instances):
                sizes = [len(gt_per_img.boxes) for gt_per_img in targets]
            else:
                sizes = [len(v["boxes"]) for v in targets]

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class ctr_HungarianMatcher(HungarianMatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def forward(self, outputs, targets, use_focal=True):
        # We flatten to compute the cost matrices in a batch
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Also concat the target labels and boxes
        if isinstance(targets[0], Instances):
            tgt_ids = torch.cat([gt_per_img.labels for gt_per_img in targets])
            tgt_bbox = torch.cat([gt_per_img.boxes for gt_per_img in targets])
        else:
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Label id conversion
        select_id = outputs["select_id"]
        tgt_ids_all = [(select_id == lid).nonzero(as_tuple=False)[0] for lid in tgt_ids]

        # We flatten to compute the cost matrices in a batch
        if use_focal:
            out_prob = outputs["pred_logits"][:, :, tgt_ids_all].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Compute the classification cost.
        if use_focal:
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                            box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        if isinstance(targets[0], Instances):
            sizes = [len(gt_per_img.boxes) for gt_per_img in targets]
        else:
            sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class QualityAwarePLMatcher(nn.Module):
    """Match pseudo labels only when their predicted geometry is reliable.

    The regular GT matcher is deliberately left untouched.  This matcher is
    used after GT matching, on the remaining queries, and puts an explicit
    dummy column beside every pseudo target.  A legal match receives a strong
    negative reward, while an IoU-ineligible target can only be assigned to a
    dummy column.  Returned query indices are in the original query space.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 1.0, cost_giou: float = 1.0, iou_min: float = 0.5):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.iou_min = float(iou_min)
        self.last_diagnostics = {"candidates": 0, "accepted": 0, "rejected_quality": 0}

    @staticmethod
    def _target_parts(target):
        if isinstance(target, Instances):
            return target.labels, target.boxes
        return target["labels"], target["boxes"]

    @staticmethod
    def _to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(-1, 4)
        return box_cxcywh_to_xyxy(boxes)

    def _one(self, output, target, available_query_indices=None):
        pred_logits = output["pred_logits"]
        pred_boxes = output.get("pred_boxes", output.get("pred_boxes_for_matching_pre"))
        select_id = output["select_id"]
        if pred_logits.dim() == 3:
            pred_logits = pred_logits[0]
        if pred_boxes.dim() == 3:
            pred_boxes = pred_boxes[0]
        labels, target_boxes = self._target_parts(target)
        labels = labels.to(device=pred_logits.device, dtype=torch.long)
        target_boxes = target_boxes.to(device=pred_boxes.device, dtype=pred_boxes.dtype)
        if available_query_indices is None:
            available_query_indices = torch.arange(pred_logits.shape[0], device=pred_logits.device, dtype=torch.long)
        else:
            available_query_indices = torch.as_tensor(available_query_indices, device=pred_logits.device, dtype=torch.long)
        if labels.numel() == 0 or available_query_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        columns = []
        for label in labels.tolist():
            matches = (select_id.to(device=pred_logits.device) == int(label)).nonzero(as_tuple=False).flatten()
            if matches.numel() == 0:
                columns.append(-1)
            else:
                columns.append(int(matches[0].item()))
        query_logits = pred_logits[available_query_indices]
        query_boxes = pred_boxes[available_query_indices]
        target_xyxy = self._to_xyxy(target_boxes)
        query_xyxy = self._to_xyxy(query_boxes)
        ious = box_iou(query_xyxy, target_xyxy)
        class_cost = query_logits.new_zeros((query_logits.shape[0], labels.shape[0]))
        valid_cols = []
        for target_index, column in enumerate(columns):
            if column < 0:
                continue
            valid_cols.append(target_index)
            probability = query_logits[:, column].sigmoid()
            alpha = 0.25
            gamma = 2.0
            neg = (1 - alpha) * probability.pow(gamma) * (-(1 - probability + 1e-8).log())
            pos = alpha * (1 - probability).pow(gamma) * (-(probability + 1e-8).log())
            class_cost[:, target_index] = pos - neg
        bbox_cost = torch.cdist(query_boxes, target_boxes, p=1)
        giou_cost = -generalized_box_iou(query_xyxy, target_xyxy)
        combined = self.cost_class * class_cost + self.cost_bbox * bbox_cost + self.cost_giou * giou_cost
        quality = ious >= self.iou_min
        if valid_cols:
            quality[:, [i for i in range(len(columns)) if i not in valid_cols]] = False
        else:
            quality.zero_()
        self.last_diagnostics = {
            "candidates": int(labels.numel()),
            "accepted": 0,
            "rejected_quality": int(labels.numel()),
        }
        # Q rows and P real target columns plus Q dummy columns.  A legal
        # target reward is lower than all dummy costs; invalid targets remain
        # expensive and therefore cannot steal a query from a dummy.
        q, p = combined.shape
        matrix = combined.new_zeros((q, p + q))
        finite_cost = combined[quality]
        if finite_cost.numel():
            lo = finite_cost.min()
            hi = finite_cost.max()
            normalized = (combined - lo) / (hi - lo).clamp(min=1e-8)
        else:
            normalized = combined
        matrix[:, :p] = torch.where(quality, normalized - float(p + 1), combined.new_full((q, p), 1e6))
        rows, cols = linear_sum_assignment(matrix.detach().cpu().numpy())
        accepted_queries = []
        accepted_targets = []
        for row, column in zip(rows.tolist(), cols.tolist()):
            if column < p and bool(quality[row, column]):
                accepted_queries.append(int(available_query_indices[row].item()))
                accepted_targets.append(int(column))
        self.last_diagnostics["accepted"] = len(accepted_queries)
        self.last_diagnostics["rejected_quality"] = max(0, int(labels.numel()) - len(accepted_targets))
        return torch.as_tensor(accepted_queries, dtype=torch.long), torch.as_tensor(accepted_targets, dtype=torch.long)

    @torch.no_grad()
    def forward(self, outputs, targets, available_query_indices=None, *, iou_min=None):
        if iou_min is not None:
            old = self.iou_min
            self.iou_min = float(iou_min)
        try:
            pred_logits = outputs["pred_logits"]
            batch = pred_logits.shape[0]
            if not isinstance(targets, (list, tuple)):
                targets = [targets]
            if available_query_indices is None:
                available_query_indices = [None] * batch
            elif torch.is_tensor(available_query_indices) and available_query_indices.dim() == 1:
                available_query_indices = [available_query_indices] * batch
            result = []
            for batch_index in range(batch):
                output = {
                    key: (value[batch_index:batch_index + 1] if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch else value)
                    for key, value in outputs.items()
                }
                result.append(self._one(output, targets[batch_index], available_query_indices[batch_index]))
            return result
        finally:
            if iou_min is not None:
                self.iou_min = old


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Small local IoU helper for normalized or pixel xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-8)


def build_matcher(args):
    return ctr_HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou
        )
