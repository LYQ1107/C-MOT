"""Replay-aligned distillation for old classes.

The three box sources consumed here use one explicit contract:

* student predictions: normalized ``cxcywh``;
* teacher predictions: normalized ``cxcywh``;
* replay ``Instances.boxes``: normalized ``cxcywh``.

Conversion to ``xyxy`` happens only inside the IoU calculation.  KD is
eligible only for old-class GT identities that have a one-to-one Hungarian
alignment on both the student and frozen teacher queries.
"""

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .supervision import bernoulli_kl


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``cxcywh`` boxes to ``xyxy`` boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    if boxes.shape[-1] != 4:
        raise ValueError("cxcywh boxes must have four coordinates")
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2].clamp(min=0)
    h = boxes[..., 3].clamp(min=0)
    return torch.stack(
        [
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h,
        ],
        dim=-1,
    )


def _xyxy_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() == 0 or right.numel() == 0:
        return left.new_zeros((left.shape[0], right.shape[0]))
    left = left.reshape(-1, 4)
    right = right.reshape(-1, 4)
    lt = torch.maximum(left[:, None, :2], right[None, :, :2])
    rb = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_l = (left[:, 2] - left[:, 0]).clamp(min=0) * (left[:, 3] - left[:, 1]).clamp(min=0)
    area_r = (right[:, 2] - right[:, 0]).clamp(min=0) * (right[:, 3] - right[:, 1]).clamp(min=0)
    return inter / (area_l[:, None] + area_r[None, :] - inter).clamp(min=1e-8)


@torch.no_grad()
def _align_queries_to_gt(
    query_boxes_cxcywh: torch.Tensor,
    gt_boxes_cxcywh: torch.Tensor,
    iou_min: float,
) -> Tuple[Dict[int, int], torch.Tensor]:
    """Return a one-to-one ``gt_index -> query_index`` Hungarian mapping."""
    query_boxes_cxcywh = query_boxes_cxcywh.reshape(-1, 4)
    gt_boxes_cxcywh = gt_boxes_cxcywh.reshape(-1, 4)
    ious = _xyxy_iou(
        _cxcywh_to_xyxy(query_boxes_cxcywh),
        _cxcywh_to_xyxy(gt_boxes_cxcywh),
    )
    if ious.numel() == 0:
        return {}, ious
    finite_ious = torch.where(torch.isfinite(ious), ious, torch.zeros_like(ious))
    rows, cols = linear_sum_assignment((-finite_ious.detach().cpu().numpy()))
    mapping: Dict[int, int] = {}
    for query_index, gt_index in zip(rows, cols):
        if float(finite_ious[int(query_index), int(gt_index)].item()) >= float(iou_min):
            mapping[int(gt_index)] = int(query_index)
    return mapping, finite_ious


class ReplayAlignedDistillation(nn.Module):
    """Old-class Bernoulli sigmoid-KL on replay GT-aligned queries."""

    def __init__(
        self,
        temperature: float = 2.0,
        iou_min: float = 0.5,
        score_min: float = 0.5,
        lambda_kd: float = 0.25,
        warmup_steps: int = 100,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.iou_min = float(iou_min)
        self.score_min = float(score_min)
        self.lambda_kd = float(lambda_kd)
        self.warmup_steps = int(warmup_steps)
        self.training_step = 0
        self.last_diagnostics: Dict[str, float] = {
            "replay_gt_objects": 0.0,
            "student_iou_pass": 0.0,
            "teacher_iou_pass": 0.0,
            "joint_alignment_pass": 0.0,
            "shared_old_class_objects": 0.0,
            "teacher_score_pass": 0.0,
            "valid_kd_objects": 0.0,
            "kd_cells": 0.0,
            "kd_pairs": 0.0,
            "effective_lambda_sum": 0.0,
            "effective_lambda_count": 0.0,
            "effective_lambda_mean": 0.0,
            "effective_lambda_last": 0.0,
        }

    def set_training_step(self, step: int) -> None:
        self.training_step = int(step)

    def effective_lambda(self) -> float:
        if self.warmup_steps <= 0:
            return self.lambda_kd
        return self.lambda_kd * min(1.0, float(self.training_step) / float(self.warmup_steps))

    @staticmethod
    def _get_field(instances, name: str, default=None):
        if instances is None or not instances.has(name):
            return default
        return getattr(instances, name)

    @staticmethod
    def _frame_value(values, index: int):
        if values is None:
            return None
        if isinstance(values, (list, tuple)):
            if not values:
                return None
            return values[min(int(index), len(values) - 1)]
        return values

    @staticmethod
    def _select_id_columns(
        student_select_ids: Optional[torch.Tensor],
        teacher_select_ids: Optional[torch.Tensor],
        old_global_ids: Sequence[int],
    ) -> Dict[int, Tuple[int, int]]:
        if student_select_ids is None or teacher_select_ids is None:
            return {}
        student_select_ids = torch.as_tensor(student_select_ids).reshape(-1).tolist()
        teacher_select_ids = torch.as_tensor(teacher_select_ids).reshape(-1).tolist()
        student_columns = {int(value): index for index, value in enumerate(student_select_ids)}
        teacher_columns = {int(value): index for index, value in enumerate(teacher_select_ids)}
        return {
            int(global_id): (student_columns[int(global_id)], teacher_columns[int(global_id)])
            for global_id in old_global_ids
            if int(global_id) in student_columns and int(global_id) in teacher_columns
        }

    def forward(
        self,
        student_outputs: Dict[str, object],
        teacher_outputs: Dict[str, object],
        replay_instances: Sequence[object],
        old_global_ids: Sequence[int],
        student_select_ids: Optional[Sequence[torch.Tensor]] = None,
        teacher_select_ids: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        student_logits = student_outputs.get("pred_logits")
        teacher_logits = teacher_outputs.get("pred_logits")
        student_boxes = student_outputs.get("pred_boxes")
        teacher_boxes = teacher_outputs.get("pred_boxes")
        if not isinstance(student_logits, (list, tuple)):
            student_logits = [student_logits]
        if not isinstance(teacher_logits, (list, tuple)):
            teacher_logits = [teacher_logits]
        if not isinstance(student_boxes, (list, tuple)):
            student_boxes = [student_boxes]
        if not isinstance(teacher_boxes, (list, tuple)):
            teacher_boxes = [teacher_boxes]

        old_ids = tuple(sorted({int(value) for value in old_global_ids}))
        total = None
        replay_count = 0
        student_iou_pass = 0
        teacher_iou_pass = 0
        joint_alignment_pass = 0
        shared_old_class_objects = 0
        teacher_score_pass = 0
        valid_count = 0
        kd_cells = 0

        for frame_index, instances in enumerate(replay_instances):
            labels = self._get_field(instances, "labels")
            sources = self._get_field(instances, "label_sources")
            boxes = self._get_field(instances, "boxes")
            if labels is None or boxes is None or len(labels) == 0:
                continue
            if sources is None:
                sources = torch.zeros_like(labels)
            old_label = torch.zeros_like(labels, dtype=torch.bool)
            for global_id in old_ids:
                old_label |= labels == int(global_id)
            legal_source = (sources == 0) | (sources == 1)
            keep = legal_source & old_label & (labels >= 0)
            if not bool(keep.any()):
                continue
            gt_labels = labels[keep]
            gt_boxes = boxes[keep].reshape(-1, 4)
            replay_count += int(gt_labels.numel())

            s_boxes = self._frame_value(student_boxes, frame_index)
            t_boxes = self._frame_value(teacher_boxes, frame_index)
            s_logits = self._frame_value(student_logits, frame_index)
            t_logits = self._frame_value(teacher_logits, frame_index)
            if any(value is None for value in (s_boxes, t_boxes, s_logits, t_logits)):
                continue
            if s_boxes.dim() == 3:
                s_boxes = s_boxes[0]
            if t_boxes.dim() == 3:
                t_boxes = t_boxes[0]
            if s_logits.dim() == 3:
                s_logits = s_logits[0]
            if t_logits.dim() == 3:
                t_logits = t_logits[0]

            student_map, _ = _align_queries_to_gt(s_boxes, gt_boxes, self.iou_min)
            teacher_map, _ = _align_queries_to_gt(t_boxes, gt_boxes, self.iou_min)
            student_iou_pass += len(student_map)
            teacher_iou_pass += len(teacher_map)
            joint = sorted(set(student_map).intersection(teacher_map))
            joint_alignment_pass += len(joint)

            s_ids = self._frame_value(student_select_ids, frame_index)
            t_ids = self._frame_value(teacher_select_ids, frame_index)
            shared_columns = self._select_id_columns(s_ids, t_ids, old_ids)
            for target_index in joint:
                if not shared_columns:
                    continue
                shared_old_class_objects += 1
                global_id = int(gt_labels[target_index].item())
                columns = shared_columns.get(global_id)
                if columns is None:
                    # The true GT class must be present for the teacher score
                    # gate, even though KD itself covers every shared old class.
                    continue
                student_query = student_map[target_index]
                teacher_query = teacher_map[target_index]
                student_column, teacher_column = columns
                if float(t_logits[teacher_query, teacher_column].sigmoid().item()) < self.score_min:
                    continue
                teacher_score_pass += 1
                student_values = []
                teacher_values = []
                for _, (s_column, t_column) in sorted(shared_columns.items()):
                    student_values.append(s_logits[student_query, s_column])
                    teacher_values.append(t_logits[teacher_query, t_column])
                student_old_logits = torch.stack(student_values)
                teacher_old_logits = torch.stack(teacher_values)
                value = bernoulli_kl(
                    student_old_logits,
                    teacher_old_logits,
                    self.temperature,
                ).mean()
                total = value if total is None else total + value
                valid_count += 1
                kd_cells += len(shared_columns)

        if total is None:
            # Keep the graph connected to the student while reporting the
            # exact gate at which replay KD became ineligible.
            reference = next(value for value in student_logits if value is not None)
            total = reference.sum() * 0.0
        total = total / float(max(valid_count, 1))
        effective_lambda = self.effective_lambda()
        total = total * effective_lambda
        self.last_diagnostics = {
            "replay_gt_objects": float(replay_count),
            "student_iou_pass": float(student_iou_pass),
            "teacher_iou_pass": float(teacher_iou_pass),
            "joint_alignment_pass": float(joint_alignment_pass),
            "shared_old_class_objects": float(shared_old_class_objects),
            "teacher_score_pass": float(teacher_score_pass),
            "valid_kd_objects": float(valid_count),
            "kd_cells": float(kd_cells),
            "kd_pairs": float(valid_count),
            "effective_lambda_sum": float(effective_lambda),
            "effective_lambda_count": 1.0,
            "effective_lambda_mean": float(effective_lambda),
            "effective_lambda_last": float(effective_lambda),
        }
        return total, dict(self.last_diagnostics)
