"""
StormSight: Physics-Aware AI for Early Cyclone Prediction
Package initialization.
"""

from .encoders import ImageEncoder, SpatioTemporalEncoder, PhysicsEncoder
from .fusion import FeatureFusion
from .model import StormSightModel
from .pipeline import PreprocessingPipeline, StormSightPredictor, ActiveLearningPipeline

__version__ = "1.0.0"
__all__ = [
    "ImageEncoder",
    "SpatioTemporalEncoder",
    "PhysicsEncoder",
    "FeatureFusion",
    "StormSightModel",
    "PreprocessingPipeline",
    "StormSightPredictor",
    "ActiveLearningPipeline",
]
