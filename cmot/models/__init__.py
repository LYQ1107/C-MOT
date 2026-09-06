"""Optional C-MOT model components."""

from .motion_prior import CategoryConditionedMotionPrior, MotionForecast

__all__ = ["CategoryConditionedMotionPrior", "MotionForecast"]
