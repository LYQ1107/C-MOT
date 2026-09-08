"""Losses that have their own normalization/audit contract."""

from .motion import MotionNLLLoss, target_velocity_from_boxes
from .replay_distillation import ReplayAlignedDistillation
from .supervision import ClassificationSupervision, LossAccumulator

__all__ = [
    "MotionNLLLoss",
    "target_velocity_from_boxes",
    "ClassificationSupervision",
    "LossAccumulator",
    "ReplayAlignedDistillation",
]
