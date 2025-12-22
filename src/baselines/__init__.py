"""
KV Cache Compression Baselines
==============================

Reference implementations for benchmarking against state-of-the-art methods.
Each implementation strictly follows the logic described in the original papers.

Implemented methods:

Eviction-based:
- H2O: Heavy-Hitter Oracle (Zhang et al., NeurIPS 2023)
- StreamingLLM: Attention Sinks (Xiao et al., ICLR 2024)
- TOVA: Token Omission Via Attention (Oren et al., 2024)
- SnapKV: Observation Window-based Compression (Li et al., NeurIPS 2024)
- PyramidKV: Layer-wise Dynamic Compression (Cai et al., 2024)

Merging-based:
- KVMerger: Gaussian Kernel Merging (Wang et al., 2024)
- WeightedKV: Attention-Weighted Value Merging (Yuan et al., ICASSP 2025)
- CaM: Cache Merging (Zhang et al., ICML 2024)
"""

from .base import CompressionConfig, CompressionMethod
from .cam import CaM, CaMConfig
from .h2o import H2O, H2OConfig
from .kvmerger import KVMerger, KVMergerConfig
from .pyramidkv import PyramidKV, PyramidKVConfig
from .snapkv import SnapKV, SnapKVConfig
from .streaming import StreamingLLM, StreamingLLMConfig
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
    "SnapKV",
    "SnapKVConfig",
    "PyramidKV",
    "PyramidKVConfig",
    # Merging-based
    "KVMerger",
    "KVMergerConfig",
    "WeightedKV",
    "WeightedKVConfig",
    "CaM",
    "CaMConfig",
]
