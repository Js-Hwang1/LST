"""
LST Training Module
===================

Training utilities for Learned Super-Token compression.

Components:
    - Trainer: Main training loop with multi-objective loss
    - LongTextDataset: Dataset for long-context training (booksum, etc.)
    - CachedTextDataset: Fast dataset from pre-tokenized files
    - TrainingConfig: Configuration dataclass
"""

from .dataset import CachedTextDataset, LongTextDataset, create_dataloaders
from .trainer import LSTTrainer

__all__ = [
    "LSTTrainer",
    "LongTextDataset",
    "CachedTextDataset",
    "create_dataloaders",
]
