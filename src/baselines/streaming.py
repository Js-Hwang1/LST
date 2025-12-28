"""
StreamingLLM: Attention Sinks (Xiao et al., ICLR 2024)
Aligned with KVCache-Factory for LongBench comparison.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .base import CompressionConfig, CompressionMethod


@dataclass
class StreamingLLMConfig(CompressionConfig):
    max_capacity_prompt: int = 512  # Total KV cache size
    window_size: int | None = None  # Recent tokens (default: max_capacity_prompt - 4)


class StreamingLLM(CompressionMethod):
    """StreamingLLM: First N sink tokens + last M recent tokens (no attention)."""

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
        query_states: Tensor | None = None,
        attention_scores: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        config = self.streaming_config

        if keys.dim() == 3:
            keys = keys.unsqueeze(1)
            values = values.unsqueeze(1)
            was_3d = True
        else:
            was_3d = False

        bsz, num_heads, seq_len, head_dim = keys.shape

        if seq_len <= config.max_capacity_prompt:
            if was_3d:
                return keys.squeeze(1), values.squeeze(1)
            return keys, values

        # KVCache-Factory default: window_size = max_capacity_prompt - 4
        window_size = config.window_size if config.window_size is not None else config.max_capacity_prompt - 4
        num_sink = config.max_capacity_prompt - window_size

        # Simply slice: first num_sink + last window_size
        k_sink = keys[:, :, :num_sink, :]
        v_sink = values[:, :, :num_sink, :]
        k_recent = keys[:, :, -window_size:, :]
        v_recent = values[:, :, -window_size:, :]

        keys_out = torch.cat([k_sink, k_recent], dim=2)
        values_out = torch.cat([v_sink, v_recent], dim=2)

        if was_3d:
            return keys_out.squeeze(1), values_out.squeeze(1)
        return keys_out, values_out
