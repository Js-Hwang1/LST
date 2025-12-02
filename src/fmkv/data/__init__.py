"""
Data Module
===========

Components for collecting training data and creating datasets:
- TrajectoryCollector: Collect KV states from model forward passes
- GradientSnapshotter: Compute and cache "true forces" for training
- ForceMatchingDataset: PyTorch Dataset for training Sidecar
"""

from fmkv.data.dataset import ForceMatchingDataset, create_dataloader
from fmkv.data.trajectory import (
    TrajectoryWindow,
    TrajectoryConfig,
    load_trajectories,
)

# Lazy import for TrajectoryCollector (needs transformers)
def __getattr__(name):
    if name == "TrajectoryCollector":
        from fmkv.data.trajectory import TrajectoryCollector
        return TrajectoryCollector
    if name == "GradientCache":
        from fmkv.data.gradient_cache import GradientCache
        return GradientCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrajectoryCollector",
    "TrajectoryWindow",
    "TrajectoryConfig",
    "load_trajectories",
    "ForceMatchingDataset",
    "create_dataloader",
    "GradientCache",
]
