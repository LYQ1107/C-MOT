"""Source-aware classification supervision for continual real-video views.

The model has a sparse ``select_id`` vector rather than a dense semantic
vocabulary.  This module keeps the masks and their denominators explicit so a
partial frame cannot silently become a background target for an unseen class.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


@dataclass
class ClassificationSupervision:
    """Per-query/per-selected-column classification targets and masks."""

    targets: torch.Tensor
    gt_mask: torch.Tensor
    gt_weights: torch.Tensor
    pl_mask: torch.Tensor
    pl_weights: torch.Tensor
    matched_iou: torch.Tensor
    trusted_gt_queries: torch.Tensor
    matched_source: torch.Tensor


class LossAccumulator:
    """Accumulate source/layer/frame numerators with exactly one denominator.

    ``register`` is intentionally idempotent for a key.  A caller may register
    the valid count while constructing a layer and later add its numerator;
    registering the same key twice never doubles the denominator.
    """

    def __init__(self) -> None:
        self.counts: Dict[Tuple[str, int, int], float] = {}
        self.numerators: Dict[Tuple[str, int, int], torch.Tensor] = {}

    def clear(self) -> None:
        self.counts.clear()
        self.numerators.clear()

    def register(self, group: str, layer: int, frame: int, count: float) -> None:
        key = (str(group), int(layer), int(frame))
        # Counts are a property of the matched target set, not of the number
        # of loss terms that happen to consume it.
        self.counts[key] = max(float(count), 1.0) if count > 0 else 0.0

    def add(self, group: str, layer: int, frame: int, numerator: torch.Tensor) -> None:
        key = (str(group), int(layer), int(frame))
        previous = self.numerators.get(key)
        self.numerators[key] = numerator if previous is None else previous + numerator

    def denominator(self, group: str, layer: int, frame: int) -> float:
        return float(self.counts.get((str(group), int(layer), int(frame)), 0.0))

    def finalize(self) -> Dict[Tuple[str, int, int], torch.Tensor]:
        result: Dict[Tuple[str, int, int], torch.Tensor] = {}
        for key, numerator in self.numerators.items():
            denominator = self.counts.get(key, 0.0)
            if denominator > 0:
                result[key] = numerator / denominator
            else:
                result[key] = numerator * 0.0
        return result

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            "%s:%d:%d" % key: {
                "count": float(self.counts.get(key, 0.0)),
                "has_numerator": key in self.numerators,
            }
            for key in sorted(set(self.counts) | set(self.numerators))
        }


def focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Return a masked focal numerator without implicit reduction."""
    targets = targets.to(dtype=logits.dtype)
    mask = mask.to(dtype=logits.dtype)
    if weights is None:
        weights = torch.ones_like(mask)
    weights = weights.to(dtype=logits.dtype)
    prob = logits.sigmoid()
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    return (loss * mask * weights).sum()


def bernoulli_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Bernoulli KL(teacher || student), with the mandated T² scaling."""
    temperature = float(temperature)
    student = student_logits / temperature
    teacher = teacher_logits.detach() / temperature
    teacher_prob = teacher.sigmoid()
    student_log_prob = torch.nn.functional.logsigmoid(student)
    student_log_not = torch.nn.functional.logsigmoid(-student)
    teacher_log_prob = torch.nn.functional.logsigmoid(teacher)
    teacher_log_not = torch.nn.functional.logsigmoid(-teacher)
    value = teacher_prob * (teacher_log_prob - student_log_prob)
    value = value + (1.0 - teacher_prob) * (teacher_log_not - student_log_not)
    return value * (temperature ** 2)
