"""
PyramidKV: Dynamic KV Cache Compression (Cai et al., 2024)
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
class PyramidKVConfig(CompressionConfig):
    max_capacity_prompt: int = 512  # Total KV cache size
    window_size: int = 8  # Observation window for attention
    kernel_size: int = 7  # Pooling kernel size
    pooling: str = "maxpool"  # 'maxpool' or 'avgpool'
    beta: int = 20  # Pyramid steepness (higher = flatter)
    num_hidden_layers: int = 32  # Total layers in model


class PyramidKV(CompressionMethod):
    """PyramidKV: SnapKV + layer-wise capacity (lower layers get more)."""

    def __init__(self, config: PyramidKVConfig):
        super().__init__(config)
        self.pyramidkv_config = config

    @property
    def name(self) -> str:
        return "PyramidKV"

    def _compute_layer_capacity(self, layer_idx: int, seq_len: int) -> int:
        """KVCache-Factory pyramid formula."""
        config = self.pyramidkv_config
        base = config.max_capacity_prompt - config.window_size

        min_num = base // config.beta
        max_num = base * 2 - min_num

        available = seq_len - config.window_size
        if max_num >= available:
            max_num = available
            min_num = base * 2 - max_num

        if config.num_hidden_layers > 1:
            steps = (max_num - min_num) // (config.num_hidden_layers - 1)
        else:
            steps = 0

        capacity = max_num - layer_idx * steps
        return max(capacity, min_num)

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        query_states: Tensor | None = None,
        attention_scores: Tensor | None = None,
        layer_idx: int = 0,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        config = self.pyramidkv_config

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

        # Layer-specific capacity
        layer_capacity = self._compute_layer_capacity(layer_idx, seq_len)

        # Two regimes based on sequence length
        capacity_threshold = (config.max_capacity_prompt - window_size) * 2
        if seq_len < capacity_threshold:
            num_to_keep = config.max_capacity_prompt - window_size
        else:
            num_to_keep = layer_capacity

        if query_states is not None:
            if query_states.dim() == 3:
                query_states = query_states.unsqueeze(1)

            # Attention from observation window
            attn_weights = torch.matmul(
                query_states[..., -window_size:, :], keys.transpose(2, 3)
            ) / math.sqrt(head_dim)

            # Causal mask
            mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(window_size, device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(window_size, 1), 0)
            attn_weights[:, :, :, -window_size:] += mask[None, None, :, :]

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(keys.dtype)
            attn_weights_sum = attn_weights[:, :, :, :-window_size].sum(dim=-2)

            # Apply pooling
            if config.pooling == "avgpool":
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=config.kernel_size, padding=config.kernel_size // 2, stride=1)
            else:
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=config.kernel_size, padding=config.kernel_size // 2, stride=1)

        elif attention_scores is not None:
            if attention_scores.dim() == 4:
                attn_cache = attention_scores.mean(dim=1).sum(dim=1)
            else:
                attn_cache = attention_scores
        else:
            attn_cache = keys[:, :, :-window_size, :].norm(dim=-1)

        # Select top-k
        indices = attn_cache.topk(num_to_keep, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past = keys[:, :, :-window_size, :].gather(dim=2, index=indices)
        v_past = values[:, :, :-window_size, :].gather(dim=2, index=indices)

        keys_out = torch.cat([k_past, keys[:, :, -window_size:, :]], dim=2)
        values_out = torch.cat([v_past, values[:, :, -window_size:, :]], dim=2)

        if was_3d:
            return keys_out.squeeze(1), values_out.squeeze(1)
        return keys_out, values_out
