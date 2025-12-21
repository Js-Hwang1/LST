"""
KV Cache Compression Baselines
==============================

Reference implementations for benchmarking against state-of-the-art methods.
Each implementation strictly follows the logic described in the original papers.

Implemented methods:

Eviction-based:
- H2O: Heavy-Hitter Oracle (Zhang et al., 2023)
- StreamingLLM: Attention Sinks (Xiao et al., 2023)
- TOVA: Token Omission Via Attention (Oren et al., 2024)

Merging-based:
- ToMe: Token Merging (Bolya et al., 2023)
- KVMerger: Gaussian Kernel Merging (Wang et al., 2024)
- WeightedKV: Attention-Weighted Value Merging (Yuan et al., 2025)
- CaM: Cache Merging (Zhang et al., ICML 2024)
"""

from .base import CompressionConfig, CompressionMethod
from .cam import CaM, CaMConfig
from .h2o import H2O, H2OConfig
from .kvmerger import KVMerger, KVMergerConfig
from .streaming import StreamingLLM, StreamingLLMConfig
from .tome import ToMe, ToMeConfig
from .tova import TOVA, TOVAConfig
from .weightedkv import WeightedKV, WeightedKVConfig

__all__ = [
    # Base
    "CompressionMethod",
    "CompressionConfig",
    # Eviction-based
    "H2O",
    "H2OConfig",
    "StreamingLLM",
    "StreamingLLMConfig",
    "TOVA",
    "TOVAConfig",
    # Merging-based
    "ToMe",
    "ToMeConfig",
    "KVMerger",
    "KVMergerConfig",
    "WeightedKV",
    "WeightedKVConfig",
    "CaM",
    "CaMConfig",
]
