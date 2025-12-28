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
    chunk_size: int = 1024  # Query chunk size for memory-efficient attention


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

            # Handle GQA: expand KV heads to match query heads if needed
            num_q_heads = query_states.shape[1]
            if num_q_heads != num_heads:
                num_kv_groups = num_q_heads // num_heads
                keys_expanded = keys.repeat_interleave(num_kv_groups, dim=1)
            else:
                keys_expanded = keys
                num_kv_groups = 1

            # Chunked attention to avoid OOM on long sequences
            # Accumulate attention sums across query chunks
            past_len = seq_len - window_size
            attn_sum = torch.zeros(bsz, num_q_heads, past_len, device=keys.device, dtype=keys.dtype)
            chunk_size = config.chunk_size
            scale = 1.0 / math.sqrt(head_dim)

            for q_start in range(0, seq_len, chunk_size):
                q_end = min(q_start + chunk_size, seq_len)
                q_chunk = query_states[:, :, q_start:q_end, :]
                chunk_len = q_end - q_start

                # Attention: (bsz, num_q_heads, chunk, seq_len)
                attn_chunk = torch.matmul(q_chunk, keys_expanded.transpose(2, 3)) * scale

                # Causal mask: query at position q_start+i can only attend to keys 0..q_start+i
                # Vectorized: row[i] masks out positions > q_start + i
                row_idx = torch.arange(chunk_len, device=keys.device).unsqueeze(1)
                col_idx = torch.arange(seq_len, device=keys.device).unsqueeze(0)
                causal_mask = col_idx > (q_start + row_idx)
                attn_chunk = attn_chunk.masked_fill(causal_mask[None, None, :, :], torch.finfo(attn_chunk.dtype).min)

                attn_chunk = F.softmax(attn_chunk, dim=-1, dtype=torch.float32).to(keys.dtype)

                # Sum attention to past tokens (exclude recent window)
                attn_sum += attn_chunk[:, :, :, :past_len].sum(dim=2)

            # For GQA, average across query head groups
            if num_q_heads != num_heads:
                attn_cache = attn_sum.view(bsz, num_heads, num_kv_groups, -1).mean(dim=2)
            else:
                attn_cache = attn_sum

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
