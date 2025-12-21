"""
LST Training Module
===================

Training utilities for Learned Super-Token compression.

Components:
    - Trainer: Main training loop with multi-objective loss
    - TextDataset: Efficient text dataset for training
    - TrainingConfig: Configuration dataclass
"""

from .trainer import LSTTrainer
from .dataset import TextDataset, create_dataloaders

__all__ = [
    "LSTTrainer",
    "TextDataset",
    "create_dataloaders",
]
