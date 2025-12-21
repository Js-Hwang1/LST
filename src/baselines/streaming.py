"""
StreamingLLM: Efficient Streaming Language Models with Attention Sinks
=======================================================================

Reference: Xiao et al. "Efficient Streaming Language Models with Attention Sinks" (2023)

Paper insight: Initial tokens ("attention sinks") accumulate attention even if
not semantically important. Keeping them stabilizes generation for infinitely
long sequences.

CRITICAL DISTINCTION from other methods:
- StreamingLLM is a SELECTION method, NOT a compression method
- It EVICTS the middle entirely, keeping only sink + recent
- There is NO compression, NO merging, NO super-tokens
- Information from evicted tokens is LOST permanently

Faithful implementation:
1. Keep first `num_sink` tokens (attention sinks)
2. Keep last `num_recent` tokens (sliding window)
3. EVICT everything in between (NO compression!)
4. Total cache = num_sink + num_recent tokens

This is fundamentally different from LST which COMPRESSES the middle
into super-tokens that retain aggregate information.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .base import CompressionConfig, CompressionMethod


@dataclass
class StreamingLLMConfig(CompressionConfig):
    """StreamingLLM-specific configuration."""

    # Note: StreamingLLM uses num_sink and num_recent from base config
    # budget = num_sink + num_recent (no middle tokens)
    pass


class StreamingLLM(CompressionMethod):
    """
    StreamingLLM: Attention Sink + Sliding Window.

    Strategy:
    - Keep first `num_sink` tokens as attention sinks
    - Keep last `num_recent` tokens as sliding window
    - EVICT all middle tokens (no compression!)

    This is faithful to the paper:
    - No token merging or compression
    - Simple concatenation of sink + recent
    - O(1) complexity per step
    """

    def __init__(self, config: StreamingLLMConfig):
        super().__init__(config)
        self.streaming_config = config

    @property
    def name(self) -> str:
        return "StreamingLLM"

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_scores: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply StreamingLLM eviction: keep sink + recent, evict middle.

        IMPORTANT: This is NOT compression - middle tokens are LOST.

        Args:
            keys: Key tensor (batch, seq_len, d_head) or (batch, n_heads, seq_len, d_head)
            values: Value tensor with same shape as keys
            attention_scores: Ignored (StreamingLLM doesn't use attention scores)

        Returns:
            Tuple of (evicted_keys, evicted_values) with middle removed
        """
        is_4d = keys.dim() == 4
        if is_4d:
            seq_len = keys.shape[2]
        else:
            seq_len = keys.shape[1]

        num_sink = self.config.num_sink
        num_recent = self.config.num_recent
        budget = num_sink + num_recent

        # If sequence fits within budget, no eviction needed
        if seq_len <= budget:
            return keys, values

        # Extract sink tokens (first num_sink)
        if is_4d:
            k_sink = keys[:, :, :num_sink, :]
            v_sink = values[:, :, :num_sink, :]
            k_recent = keys[:, :, -num_recent:, :]
            v_recent = values[:, :, -num_recent:, :]
            # Concatenate sink + recent (EVICT middle)
            keys_out = torch.cat([k_sink, k_recent], dim=2)
            values_out = torch.cat([v_sink, v_recent], dim=2)
        else:
            k_sink = keys[:, :num_sink, :]
            v_sink = values[:, :num_sink, :]
            k_recent = keys[:, -num_recent:, :]
            v_recent = values[:, -num_recent:, :]
            # Concatenate sink + recent (EVICT middle)
            keys_out = torch.cat([k_sink, k_recent], dim=1)
            values_out = torch.cat([v_sink, v_recent], dim=1)

        return keys_out, values_out


def streaming_evict(
    keys: Tensor,
    values: Tensor,
    num_sink: int = 4,
    num_recent: int = 256,
) -> tuple[Tensor, Tensor]:
    """
    Standalone StreamingLLM eviction function for testing and integration.

    Args:
        keys: Key tensor
        values: Value tensor
        num_sink: Number of initial tokens to keep (attention sinks)
        num_recent: Number of recent tokens to keep (sliding window)

    Returns:
        Tuple of (evicted_keys, evicted_values)
    """
    config = StreamingLLMConfig(num_sink=num_sink, num_recent=num_recent)
    method = StreamingLLM(config)
    return method.compress(keys, values)
