"""Causal motion mixture loss with one explicit normalization point."""

import math
from typing import Optional

import torch
from torch import nn

from ..models.motion_prior import MotionForecast, boxes_to_state


def target_velocity_from_boxes(
    current_boxes: torch.Tensor,
    next_boxes: torch.Tensor,
    delta_t: torch.Tensor,
) -> torch.Tensor:
    """Build state velocity from two legal same-object observations."""

    dt = torch.as_tensor(delta_t, device=current_boxes.device, dtype=torch.float32)
    if dt.dim() == 0:
        dt = dt.expand(current_boxes.shape[0])
    return (boxes_to_state(next_boxes) - boxes_to_state(current_boxes)) / dt[:, None].clamp(min=1.0e-6)


class MotionNLLLoss(nn.Module):
    """Mixture diagonal-Gaussian NLL over valid trajectory pairs.

    The returned value is normalized exactly once by the weighted valid-pair
    count.  ``last_sum`` and ``last_count`` are detached audit values, not a
    second normalization path.
    """

    def __init__(self, min_log_std: float = -5.0, max_log_std: float = 3.0):
        super().__init__()
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.last_sum: Optional[float] = None
        self.last_count: int = 0

    def forward(
        self,
        forecast: MotionForecast,
        next_targets: torch.Tensor,
        pair_valid: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        means = forecast.velocity_mean.float()
        log_std = forecast.velocity_log_std.float().clamp(self.min_log_std, self.max_log_std)
        mixture = forecast.mixture_logits.float()
        targets = next_targets.float()
        valid = pair_valid.to(device=means.device, dtype=torch.bool)
        sample_weights = weights.to(device=means.device, dtype=means.dtype)
        if means.dim() != 3 or means.shape[-1] != 4:
            raise ValueError("forecast velocity_mean must have shape [N, K, 4]")
        if targets.shape != (means.shape[0], 4):
            raise ValueError("next_targets must have shape [N, 4]")
        if valid.shape != (means.shape[0],) or sample_weights.shape != (means.shape[0],):
            raise ValueError("pair_valid and weights must have shape [N]")

        variance_term = (targets[:, None, :] - means) / log_std.exp()
        log_normal = -0.5 * (
            variance_term.square() + 2.0 * log_std + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_prob = torch.log_softmax(mixture, dim=-1) + log_normal
        per_pair = -torch.logsumexp(log_prob, dim=-1)
        effective_weights = sample_weights.clamp(min=0.0) * valid.to(means.dtype)
        weighted_sum = (per_pair * effective_weights).sum()
        count = effective_weights.sum()
        self.last_sum = float(weighted_sum.detach().item())
        self.last_count = int(valid.sum().detach().item())
        if float(count.detach().item()) <= 0.0:
            return means.sum() * 0.0
        return weighted_sum / count.clamp(min=1.0e-6)
