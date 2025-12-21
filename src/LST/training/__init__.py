"""
LST Training Module
===================

Training utilities for Learned Super-Token compression.

Components:
    - Trainer: Main training loop with multi-objective loss
    - TextDataset: Efficient text dataset (on-the-fly tokenization)
    - CachedTextDataset: Fast dataset from pre-tokenized files
    - TrainingConfig: Configuration dataclass
"""

from .dataset import CachedTextDataset, TextDataset, create_dataloaders
from .trainer import LSTTrainer

__all__ = [
    "LSTTrainer",
    "TextDataset",
    "CachedTextDataset",
    "create_dataloaders",
]
