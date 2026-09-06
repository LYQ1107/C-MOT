"""Explicit updater/context boundary for category-conditioned dynamics."""

from dataclasses import dataclass
import hashlib
from typing import Optional

import torch
from torch import nn

from .motion_prior import CategoryConditionedMotionPrior, MotionForecast


try:  # The OVTR checkout is added to sys.path by the runtime before use.
    from updater import Category_Information_Propagator as _CIPBase
except ImportError:  # pragma: no cover - exercised by standalone API tests
    class _CIPBase(nn.Module):
        """Minimal legacy-compatible fallback for importing the public API."""

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, data):
            return data.get("track_instances")


@dataclass
class MotionContext:
    forecast: Optional[MotionForecast]
    stable_uids: torch.Tensor
    current_timestamp: Optional[float]
    predicted_current_boxes: torch.Tensor
    valid: torch.Tensor
    semantic_column_fingerprint: str
    next_delta_t: torch.Tensor


def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1.0e-6, 1.0 - 1.0e-6)
    return torch.log(value / (1.0 - value))


class CategoryDynamicsUpdater(_CIPBase):
    """CIP-compatible updater that can return an explicit MotionContext.

    The ordinary ``forward`` signature still returns ``Instances``.  New
    callers use ``update_with_context`` and receive the context used for
    supervision without relying on a mutable ``last_context`` cache.
    """

    def __init__(self, *args, motion_prior=None, semantic_column_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_prior = motion_prior
        self.semantic_column_ids = tuple(int(v) for v in (semantic_column_ids or ()))

    def _fingerprint(self) -> str:
        payload = ",".join(str(v) for v in self.semantic_column_ids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def update_with_context(self, data, frame_context=None):
        merged = super().forward(data)
        frame_context = frame_context or {}
        if self.motion_prior is None:
            empty_device = next(self.parameters(), torch.zeros(1)).device
            empty = torch.empty((0,), dtype=torch.long, device=empty_device)
            return merged, MotionContext(
                forecast=None,
                stable_uids=empty,
                current_timestamp=frame_context.get("timestamp"),
                predicted_current_boxes=torch.empty((0, 4), device=empty_device),
                valid=torch.empty((0,), dtype=torch.bool, device=empty_device),
                semantic_column_fingerprint=self._fingerprint(),
                next_delta_t=torch.empty((0,), device=empty_device),
            )

        history_boxes = frame_context.get("history_boxes")
        history_times = frame_context.get("history_times")
        history_valid = frame_context.get("history_valid")
        history_quality = frame_context.get("history_quality")
        semantic_context = frame_context.get("semantic_context")
        next_delta_t = frame_context.get("next_delta_t")
        if any(value is None for value in (history_boxes, history_times, history_valid, history_quality, next_delta_t)):
            raise ValueError("motion frame_context is missing a causal history field")
        forecast = self.motion_prior(
            history_boxes,
            history_times,
            history_valid,
            history_quality,
            semantic_context,
            next_delta_t,
        )
        if merged is not None and hasattr(merged, "pred_boxes") and len(merged) == len(forecast.next_boxes):
            base_reference = _inverse_sigmoid(merged.pred_boxes.detach().clone())
            merged.ref_pts = torch.where(forecast.valid[:, None], _inverse_sigmoid(forecast.next_boxes), base_reference)
        if merged is not None and hasattr(merged, "obj_idxes"):
            stable_uids = merged.obj_idxes.detach().clone()
        else:
            stable_uids = torch.arange(forecast.next_boxes.shape[0], device=forecast.next_boxes.device)
        predicted_current = frame_context.get("predicted_current_boxes")
        if predicted_current is None:
            predicted_current = history_boxes[:, -1]
        return merged, MotionContext(
            forecast=forecast,
            stable_uids=stable_uids,
            current_timestamp=frame_context.get("timestamp"),
            predicted_current_boxes=predicted_current,
            valid=forecast.valid,
            semantic_column_fingerprint=self._fingerprint(),
            next_delta_t=torch.as_tensor(next_delta_t, device=forecast.next_boxes.device).reshape(-1),
        )

    def forward(self, data, frame_context=None, id_gt=None):
        merged, _ = self.update_with_context(data, frame_context=frame_context)
        return merged
