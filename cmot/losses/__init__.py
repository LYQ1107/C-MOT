"""Losses that have their own normalization/audit contract."""

from .motion import MotionNLLLoss, target_velocity_from_boxes

__all__ = ["MotionNLLLoss", "target_velocity_from_boxes"]
