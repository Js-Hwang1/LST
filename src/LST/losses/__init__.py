"""
LST Training Losses
===================

Novel training objectives for Learned Super-Token compression.

Key insight: Direct PPL training only optimizes for specific continuations,
but compressed caches will serve arbitrary future queries. We need losses
that ensure super-tokens work for ANY query, not just training queries.

Losses:
    - PPLLoss: Standard perplexity (for generation quality)
    - QueryProbingLoss: Attention alignment via random query probing (NOVEL)
    - DiversityLoss: Contrastive loss to prevent super-token collapse
"""

from .combined import CombinedLoss, LossWeights
from .diversity import DiversityLoss
from .ppl import PPLLoss
from .query_probing import QueryProbingLoss

__all__ = [
    "PPLLoss",
    "QueryProbingLoss",
    "DiversityLoss",
    "CombinedLoss",
    "LossWeights",
]
