"""Matched old-class replay distillation.

Only replay GT identities are eligible.  Teacher inference is intentionally
kept outside the student's autograd graph and is evaluated independently from
the student's current prediction queries.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .supervision import bernoulli_kl


def _xyxy_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() == 0 or right.numel() == 0:
        return left.new_zeros((left.shape[0], right.shape[0]))
    lt = torch.maximum(left[:, None, :2], right[None, :, :2])
    rb = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_l = (left[:, 2] - left[:, 0]).clamp(min=0) * (left[:, 3] - left[:, 1]).clamp(min=0)
    area_r = (right[:, 2] - right[:, 0]).clamp(min=0) * (right[:, 3] - right[:, 1]).clamp(min=0)
    return inter / (area_l[:, None] + area_r[None, :] - inter).clamp(min=1e-8)


def _boxes_from_prediction(boxes: torch.Tensor) -> torch.Tensor:
    # OVTR predicts normalized cxcywh for its public output.  Accept xyxy too
    # when a caller explicitly supplies a replay-side tensor.
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    cxcy = boxes[..., :2]
    wh = boxes[..., 2:].clamp(min=0)
    return torch.cat((cxcy - 0.5 * wh, cxcy + 0.5 * wh), dim=-1)


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
            "valid_kd_objects": 0.0,
            "kd_pairs": 0.0,
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

    @torch.no_grad()
    def _greedy_alignment(
        self,
        student_boxes: torch.Tensor,
        teacher_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> List[Tuple[int, int, int]]:
        if gt_boxes.numel() == 0:
            return []
        student_iou = _xyxy_iou(_boxes_from_prediction(student_boxes), gt_boxes)
        teacher_iou = _xyxy_iou(_boxes_from_prediction(teacher_boxes), gt_boxes)
        result: List[Tuple[int, int, int]] = []
        used = set()
        # Replay identities are scarce; assign the best unused student query
        # to each GT identity, then let score/IoU gates decide validity.
        for target_index in range(gt_boxes.shape[0]):
            values = student_iou[:, target_index]
            if values.numel() == 0:
                continue
            order = torch.argsort(values, descending=True).tolist()
            for query_index in order:
                if int(query_index) not in used:
                    used.add(int(query_index))
                    if float(values[query_index]) >= self.iou_min and float(teacher_iou[:, target_index].max()) >= self.iou_min:
                        teacher_query = int(torch.argmax(teacher_iou[:, target_index]).item())
                        result.append((int(query_index), teacher_query, target_index))
                    break
        return result

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
        total = None
        replay_count = valid_count = 0
        old_ids = {int(value) for value in old_global_ids}
        for frame_index, instances in enumerate(replay_instances):
            labels = self._get_field(instances, "labels")
            sources = self._get_field(instances, "label_sources")
            boxes = self._get_field(instances, "boxes")
            if labels is None or boxes is None or len(labels) == 0:
                continue
            if sources is None:
                sources = torch.zeros_like(labels)
            # Both source_gt and a previously saved gt_replay snapshot are
            # legal memory supervision.  Current PL is intentionally excluded
            # from KD and cannot become a replay identity.
            keep = torch.isin(sources, torch.as_tensor((0, 1), device=sources.device)) & torch.tensor(
                [int(v) in old_ids for v in labels.tolist()], device=labels.device
            )
            keep &= labels >= 0
            if not bool(keep.any()):
                continue
            gt_labels = labels[keep]
            gt_boxes = boxes[keep]
            replay_count += int(gt_labels.numel())
            s_boxes = student_boxes[min(frame_index, len(student_boxes) - 1)]
            t_boxes = teacher_boxes[min(frame_index, len(teacher_boxes) - 1)]
            s_logits = student_logits[min(frame_index, len(student_logits) - 1)]
            t_logits = teacher_logits[min(frame_index, len(teacher_logits) - 1)]
            if s_boxes.dim() == 3:
                s_boxes = s_boxes[0]
            if t_boxes.dim() == 3:
                t_boxes = t_boxes[0]
            if s_logits.dim() == 3:
                s_logits = s_logits[0]
            if t_logits.dim() == 3:
                t_logits = t_logits[0]
            pairs = self._greedy_alignment(s_boxes, t_boxes, gt_boxes)
            s_ids = student_select_ids[frame_index] if student_select_ids is not None else None
            t_ids = teacher_select_ids[frame_index] if teacher_select_ids is not None else None
            for student_query, teacher_query, target_pos in pairs:
                global_id = int(gt_labels[target_pos].item())
                if s_ids is None or t_ids is None:
                    continue
                s_matches = (s_ids == global_id).nonzero(as_tuple=False).flatten()
                t_matches = (t_ids == global_id).nonzero(as_tuple=False).flatten()
                if s_matches.numel() == 0 or t_matches.numel() == 0:
                    continue
                s_col = int(s_matches[0].item())
                t_col = int(t_matches[0].item())
                if float(t_logits[teacher_query, t_col].sigmoid()) < self.score_min:
                    continue
                value = bernoulli_kl(
                    s_logits[student_query, s_col],
                    t_logits[teacher_query, t_col],
                    self.temperature,
                )
                total = value if total is None else total + value
                valid_count += 1
        if total is None:
            # Keep the graph connected to the current model while reporting
            # equivalence to replay without valid KD objects.
            reference = student_logits[0]
            total = reference.sum() * 0.0
        total = total / float(max(valid_count, 1))
        total = total * self.effective_lambda()
        self.last_diagnostics = {
            "replay_gt_objects": float(replay_count),
            "valid_kd_objects": float(valid_count),
            "kd_pairs": float(valid_count),
            "effective_lambda": float(self.effective_lambda()),
        }
        return total, dict(self.last_diagnostics)
