"""
LST Sidecar Network
===================

The core compression network that maps windows of KV pairs
to compact super-tokens.

Components:
    - Encoder: Captures intra-window dependencies (Transformer/GIN/MLP)
    - Aggregator: Compresses N tokens to 1 (Set Transformer/Attention/Mean)
    - Network: Complete sidecar with hard norm projection
"""

from .network import Sidecar, SidecarPPL

__all__ = [
    "Sidecar",
    "SidecarPPL",
]
