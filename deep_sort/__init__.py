"""
DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)
Package for object tracking with Kalman Filtering and appearance matching.
"""

from .tracker import DeepSortTracker
from .detection import Detection

__all__ = ["DeepSortTracker", "Detection"]

