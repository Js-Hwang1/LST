"""
LST: Learned Super-Token KV Cache Compression
==============================================

A novel approach to KV cache compression that learns to compress windows
of KV pairs into compact super-tokens while preserving language modeling quality.

Key idea: Instead of evicting tokens (H2O, StreamingLLM) or simple merging (ToMe),
LST trains a small sidecar network to produce optimal compressed representations.

Novel Training Objective (QPAA - Query-Probing Attention Alignment):
    - PPL alone optimizes for specific continuations
    - QPAA ensures super-tokens work for ANY future query
    - Random probe queries approximate the full query distribution

Architecture:
    [Window of KV pairs] -> [Encoder] -> [Aggregator] -> [Super-Token]
    (batch, N, 2*d)         (Transformer)  (Set Attention)  (batch, 1, 2*d)

Modules:
    - sidecar: Core compression network (Encoder + Aggregator + Projection)
    - losses: Training objectives (PPL, QPAA, Diversity)
    - training: Training utilities and trainer class
"""

from .config import LSTConfig, TrainingConfig
from .sidecar import Sidecar, SidecarPPL

__all__ = [
    "LSTConfig",
    "TrainingConfig",
    "Sidecar",
    "SidecarPPL",
]
