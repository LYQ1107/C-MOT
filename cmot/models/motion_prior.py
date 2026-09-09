"""Small causal motion prior used by the C-MOT protocol.

The OVTR adapter currently keeps its legacy one-step head for checkpoint
compatibility.  This module is the explicit, independently testable motion
prior interface from the protocol: it consumes only observations at or
before the current frame and returns a multimodal forecast in normalized box
coordinates.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


_EPS = 1.0e-6


@dataclass
class MotionForecast:
    """The tensors returned by :class:`CategoryConditionedMotionPrior`."""

    velocity_mean: torch.Tensor
    velocity_log_std: torch.Tensor
    mixture_logits: torch.Tensor
    next_boxes: torch.Tensor
    valid: torch.Tensor

    def __getitem__(self, key):
        return getattr(self, key)


def boxes_to_state(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``cx, cy, w, h`` boxes to the model state."""

    boxes = boxes.float()
    # Keep the conversion purely functional.  Slicing a tensor that still
    # participates in the motion-loss graph and assigning through that view
    # can invalidate autograd's saved version counter on real track batches.
    return torch.cat(
        [boxes[..., :2], torch.log(boxes[..., 2:].clamp(min=_EPS))], dim=-1
    )


def state_to_boxes(state: torch.Tensor) -> torch.Tensor:
    """Convert the motion state back to bounded normalized boxes."""

    # As above, avoid in-place writes through ``[..., :2]``/``[..., 2:]``.
    return torch.cat(
        [state[..., :2].clamp(0.0, 1.0), state[..., 2:].exp().clamp(1.0e-4, 1.0)],
        dim=-1,
    )


def normalized_class_evidence(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return a fixed-rule routing context, not a calibrated probability."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = (logits.float() / float(temperature)).sigmoid().clamp(min=0.0)
    return scores / scores.sum(dim=-1, keepdim=True).clamp(min=_EPS)


class CategoryConditionedMotionPrior(nn.Module):
    """Causal shared-mixture motion prior with an agnostic ablation.

    ``history_boxes`` has shape ``[N, L, 4]`` and uses normalized
    ``cx, cy, w, h`` coordinates.  Times are seconds.  Invalid history rows
    do not update the GRU state, and non-positive/missing time intervals are
    not treated as motion observations.  ``semantic_context`` is expected to
    be built from frozen text embeddings and current predictions by the
    caller; it is detached here so the motion loss cannot change class
    routing as a shortcut.
    """

    def __init__(
        self,
        history_length: int = 4,
        hidden_dim: int = 128,
        num_modes: int = 3,
        semantic_dim: int = 512,
        mode: str = "class_conditioned",
        velocity_limit: float = 1.25,
        min_history_points: int = 2,
    ):
        super().__init__()
        if history_length < 1 or hidden_dim < 1 or num_modes < 1:
            raise ValueError("history_length, hidden_dim and num_modes must be positive")
        if float(velocity_limit) <= 0:
            raise ValueError("velocity_limit must be positive")
        if int(min_history_points) < 1:
            raise ValueError("min_history_points must be at least one")
        mode = {"category_conditioned": "class_conditioned", "category_agnostic": "class_agnostic"}.get(mode, mode)
        if mode not in ("class_conditioned", "class_agnostic"):
            raise ValueError("mode must be class_conditioned or class_agnostic")
        self.history_length = int(history_length)
        self.hidden_dim = int(hidden_dim)
        self.num_modes = int(num_modes)
        self.semantic_dim = int(semantic_dim)
        self.mode = mode
        self.velocity_limit = float(velocity_limit)
        self.min_history_points = int(min_history_points)

        # state (4), causal difference (4), quality (1), valid (1), delta-t (1)
        self.history_input = nn.Linear(11, 64)
        self.history_gru = nn.GRUCell(64, self.hidden_dim)
        self.semantic_projection = nn.Sequential(
            nn.Linear(self.semantic_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        fusion_dim = self.hidden_dim + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.velocity_head = nn.Linear(self.hidden_dim, self.num_modes * 4)
        self.log_std_head = nn.Linear(self.hidden_dim, self.num_modes * 4)
        self.mixture_head = nn.Linear(self.hidden_dim, self.num_modes)
        # A shared semantic residual has the same capacity in the agnostic
        # control; only its semantic input is zeroed in that mode.
        self.semantic_residual = nn.Linear(self.hidden_dim, 4)

        # The final prediction layers start at a valid no-motion fallback,
        # while preceding layers remain trainable and non-zero.
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)
        nn.init.zeros_(self.semantic_residual.weight)
        nn.init.zeros_(self.semantic_residual.bias)
        nn.init.zeros_(self.mixture_head.weight)
        nn.init.zeros_(self.mixture_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def _prepare_semantic(self, semantic_context: Optional[torch.Tensor], n: int, device, dtype) -> torch.Tensor:
        if semantic_context is None:
            context = torch.zeros((n, self.semantic_dim), device=device, dtype=dtype)
        else:
            context = semantic_context
            if context.dim() == 3:
                context = context[:, -1]
            if context.shape != (n, self.semantic_dim):
                raise ValueError(
                    "semantic_context must have shape [N, semantic_dim], got %s" % (tuple(context.shape),)
                )
            context = context.detach().to(device=device, dtype=dtype)
        if self.mode == "class_agnostic":
            context = torch.zeros_like(context)
        return context

    def forward(
        self,
        history_boxes: torch.Tensor,
        history_times: torch.Tensor,
        history_valid: torch.Tensor,
        history_quality: torch.Tensor,
        semantic_context: Optional[torch.Tensor],
        next_delta_t: torch.Tensor,
    ) -> MotionForecast:
        if history_boxes.dim() != 3 or history_boxes.shape[-1] != 4:
            raise ValueError("history_boxes must have shape [N, L, 4]")
        n, length, _ = history_boxes.shape
        if history_times.shape != (n, length):
            raise ValueError("history_times must have shape [N, L]")
        if history_valid.shape != (n, length):
            raise ValueError("history_valid must have shape [N, L]")
        if history_quality.shape != (n, length):
            raise ValueError("history_quality must have shape [N, L]")
        if length > self.history_length:
            history_boxes = history_boxes[:, -self.history_length:]
            history_times = history_times[:, -self.history_length:]
            history_valid = history_valid[:, -self.history_length:]
            history_quality = history_quality[:, -self.history_length:]
            length = self.history_length

        device = history_boxes.device
        dtype = history_boxes.dtype if history_boxes.is_floating_point() else torch.float32
        boxes = history_boxes.to(dtype=dtype)
        times = history_times.to(device=device, dtype=dtype)
        valid = history_valid.to(device=device, dtype=torch.bool)
        quality = history_quality.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        states = boxes_to_state(boxes)

        hidden = torch.zeros((n, self.hidden_dim), device=device, dtype=dtype)
        previous_state = torch.zeros((n, 4), device=device, dtype=dtype)
        previous_time = torch.zeros((n,), device=device, dtype=dtype)
        has_observation = torch.zeros((n,), device=device, dtype=torch.bool)
        valid_history_count = torch.zeros((n,), device=device, dtype=torch.long)
        last_state = torch.zeros((n, 4), device=device, dtype=dtype)

        for index in range(length):
            current_state = states[:, index]
            current_time = times[:, index]
            finite_time = torch.isfinite(current_time)
            delta_t = current_time - previous_time
            time_ok = (~has_observation) | (finite_time & torch.isfinite(delta_t) & (delta_t > 0))
            update = valid[:, index] & finite_time & time_ok
            difference = current_state - previous_state
            difference = torch.where(update[:, None], difference, torch.zeros_like(difference))
            safe_dt = torch.where(update, delta_t.clamp(min=0.0), torch.zeros_like(delta_t))
            features = torch.cat(
                [
                    current_state,
                    difference,
                    quality[:, index:index + 1],
                    valid[:, index:index + 1].to(dtype),
                    safe_dt[:, None],
                ],
                dim=-1,
            )
            candidate = self.history_gru(torch.tanh(self.history_input(features)), hidden)
            hidden = torch.where(update[:, None], candidate, hidden)
            last_state = torch.where(update[:, None], current_state, last_state)
            previous_state = torch.where(update[:, None], current_state, previous_state)
            previous_time = torch.where(update, current_time, previous_time)
            has_observation = has_observation | update
            valid_history_count = valid_history_count + update.to(torch.long)

        context = self._prepare_semantic(semantic_context, n, device, dtype)
        fused = self.fusion(torch.cat([hidden, self.semantic_projection(context)], dim=-1))
        velocity_mean = self.velocity_head(fused).reshape(n, self.num_modes, 4)
        velocity_mean = velocity_mean + self.semantic_residual(fused)[:, None, :]
        velocity_mean = self.velocity_limit * torch.tanh(velocity_mean)
        velocity_log_std = self.log_std_head(fused).reshape(n, self.num_modes, 4).clamp(-5.0, 3.0)
        mixture_logits = self.mixture_head(fused)

        next_dt = torch.as_tensor(next_delta_t, device=device, dtype=dtype)
        if next_dt.dim() == 0:
            next_dt = next_dt.expand(n)
        if next_dt.shape != (n,):
            raise ValueError("next_delta_t must be scalar or shape [N]")
        next_dt_valid = torch.isfinite(next_dt) & (next_dt > 0)
        # Rows without enough history carry an infinite sentinel from the
        # caller's ``last_valid_time`` reduction.  Masking only at the final
        # output is unsafe: ``0 * inf`` creates NaNs in the unselected branch
        # and can still poison parameter gradients.  Use a finite zero for
        # the arithmetic and retain the boolean validity separately.
        safe_next_dt = torch.where(next_dt_valid, next_dt, torch.zeros_like(next_dt))
        forecast_valid = (valid_history_count >= self.min_history_points) & next_dt_valid
        if self.num_modes == 1:
            selected_velocity = velocity_mean[:, 0]
        else:
            selected_mode = mixture_logits.argmax(dim=-1)
            selected_velocity = velocity_mean[torch.arange(n, device=device), selected_mode]
        predicted_state = last_state + selected_velocity * safe_next_dt[:, None]
        predicted_boxes = state_to_boxes(predicted_state)
        fallback = state_to_boxes(last_state)
        next_boxes = torch.where(forecast_valid[:, None], predicted_boxes, fallback)

        return MotionForecast(
            velocity_mean=velocity_mean,
            velocity_log_std=velocity_log_std,
            mixture_logits=mixture_logits,
            next_boxes=next_boxes,
            valid=forecast_valid,
        )
