"""
H2O: Heavy-Hitter Oracle (Zhang et al., NeurIPS 2023)
Aligned with KVCache-Factory for LongBench comparison.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionConfig, CompressionMethod


@dataclass
class H2OConfig(CompressionConfig):
    max_capacity_prompt: int = 512  # Total KV cache size
    window_size: int = 8  # Recent tokens to preserve


class H2O(CompressionMethod):
    """H2O: Attention from ALL queries, NO pooling, top-k selection."""

    def __init__(self, config: H2OConfig):
        super().__init__(config)
        self.h2o_config = config

    @property
    def name(self) -> str:
        return "H2O"

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        query_states: Tensor | None = None,
        attention_scores: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        config = self.h2o_config

        if keys.dim() == 3:
            keys = keys.unsqueeze(1)
            values = values.unsqueeze(1)
            was_3d = True
        else:
            was_3d = False

        bsz, num_heads, seq_len, head_dim = keys.shape
        window_size = config.window_size

        if seq_len <= config.max_capacity_prompt:
            if was_3d:
                return keys.squeeze(1), values.squeeze(1)
            return keys, values

        if query_states is not None:
            if query_states.dim() == 3:
                query_states = query_states.unsqueeze(1)

            # Attention from ALL queries to ALL keys
            attn_weights = torch.matmul(query_states, keys.transpose(2, 3)) / math.sqrt(head_dim)

            # Full causal mask (KVCache-Factory applies to entire matrix)
            mask = torch.full((seq_len, seq_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(seq_len, device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(seq_len, 1), 0)
            attn_weights = attn_weights + mask[None, None, :, :]

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(keys.dtype)

            # Sum across ALL queries, exclude recent window from selection
            attn_cache = attn_weights[:, :, :, :-window_size].sum(dim=-2)

        elif attention_scores is not None:
            if attention_scores.dim() == 4:
                attn_cache = attention_scores.mean(dim=1).sum(dim=1)
            else:
                attn_cache = attention_scores
        else:
            # Fallback: key norms (exclude recent window)
            attn_cache = keys[:, :, :-window_size, :].norm(dim=-1)

        # Select top-k
        num_to_keep = config.max_capacity_prompt - window_size
        indices = attn_cache.topk(num_to_keep, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past = keys[:, :, :-window_size, :].gather(dim=2, index=indices)
        v_past = values[:, :, :-window_size, :].gather(dim=2, index=indices)

        keys_out = torch.cat([k_past, keys[:, :, -window_size:, :]], dim=2)
        values_out = torch.cat([v_past, values[:, :, -window_size:, :]], dim=2)

        if was_3d:
            return keys_out.squeeze(1), values_out.squeeze(1)
        return keys_out, values_out
