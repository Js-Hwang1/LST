"""
PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
=================================================================================

Reference: Cai et al. "PyramidKV: Dynamic KV Cache Compression based on
Pyramidal Information Funneling" (arXiv:2406.02069, 2024)

Paper insight: LLMs aggregate information through "Pyramidal Information Funneling":
- Lower layers: Attention scatters widely, capturing broad context
- Middle layers: Attention consolidates within specific contexts
- Upper layers: Attention focuses on critical tokens (attention sinks)

Algorithm:
1. Allocate different KV cache budgets per layer (pyramid shape)
   - Lower layers: larger budget (attention scattered)
   - Upper layers: smaller budget (attention focused on sinks)
2. Within each layer, select important KV vectors based on attention scores
3. Always preserve a window of recent tokens

Faithful implementation based on:
- Paper: https://arxiv.org/abs/2406.02069
- Official repo: https://github.com/Zefan-Cai/KVCache-Factory
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionConfig, CompressionMethod


@dataclass
class PyramidKVConfig(CompressionConfig):
    """PyramidKV-specific configuration matching official implementation."""

    # Total number of layers in the model
    num_layers: int = 32

    # Maximum capacity per layer (official default: 2048)
    # This is the BASE capacity that gets modified by pyramid allocation
    max_capacity_prompt: int = 2048

    # Window size for recent tokens always preserved (official default: 32)
    window_size: int = 32

    # Beta parameter controlling pyramid steepness (official default: 20)
    # Higher beta = flatter pyramid (more uniform allocation)
    # Lower beta = steeper pyramid (more difference between layers)
    beta: int = 20

    # Kernel size for attention pooling (official default: 5)
    kernel_size: int = 5

    # Pooling method: 'avgpool' or 'maxpool'
    pooling: str = "avgpool"


class PyramidKV(CompressionMethod):
    """
    PyramidKV: Layer-wise Dynamic KV Cache Compression.

    This implementation follows the official KVCache-Factory code:
    - Uses beta parameter to control pyramid allocation
    - Allocates more cache to lower layers, less to upper layers
    - Uses attention-based token selection with pooling
    - Always preserves a window of recent tokens
    """

    def __init__(self, config: PyramidKVConfig):
        super().__init__(config)
        self.pyramid_config = config

    @property
    def name(self) -> str:
        return "PyramidKV"

    def compute_layer_capacity(self, layer_idx: int) -> int:
        """
        Compute the KV cache capacity for a specific layer.

        Uses the official PyramidKV formula:
        - min_num = (max_capacity_prompt - window_size) // beta
        - max_num = (max_capacity_prompt - window_size) * 2 - min_num
        - steps = (max_num - min_num) // (num_layers - 1)
        - capacity = max_num - layer_idx * steps

        Args:
            layer_idx: Index of the current layer (0-indexed)

        Returns:
            Number of KV pairs to retain for this layer
        """
        config = self.pyramid_config
        base = config.max_capacity_prompt - config.window_size

        if base <= 0:
            return config.max_capacity_prompt

        min_num = base // config.beta
        max_num = base * 2 - min_num

        if config.num_layers > 1:
            steps = (max_num - min_num) // (config.num_layers - 1)
        else:
            steps = 0

        # Layer 0 gets max_num, last layer gets closer to min_num
        capacity = max_num - layer_idx * steps

        # Add back window_size and ensure minimum
        capacity = max(capacity, min_num) + config.window_size

        return int(capacity)

    def compute_attention_scores(
        self,
        keys: Tensor,
        queries: Tensor | None = None,
    ) -> Tensor:
        """
        Compute attention-based importance scores for tokens.

        If queries provided, compute attention between queries and keys.
        Otherwise, use key norms as a proxy for importance.

        Args:
            keys: (batch, seq_len, d_model)
            queries: Optional query tensor for attention computation

        Returns:
            Importance scores (batch, seq_len)
        """
        if queries is not None and queries.numel() > 0:
            # Compute attention scores
            # queries: (batch, q_len, d_model)
            # keys: (batch, k_len, d_model)
            attn = torch.matmul(queries, keys.transpose(-1, -2))
            attn = attn / (keys.shape[-1] ** 0.5)
            attn = F.softmax(attn, dim=-1)
            # Sum attention over query positions
            importance = attn.sum(dim=1)  # (batch, k_len)
        else:
            # Use key norms as importance proxy
            importance = keys.norm(dim=-1)

        return importance

    def apply_pooling(self, importance: Tensor) -> Tensor:
        """
        Apply pooling to importance scores (official PyramidKV uses this).

        Args:
            importance: (batch, seq_len)

        Returns:
            Pooled importance scores
        """
        config = self.pyramid_config
        kernel_size = config.kernel_size

        if kernel_size <= 1:
            return importance

        # Add channel dimension for pooling
        importance = importance.unsqueeze(1)  # (batch, 1, seq_len)

        if config.pooling == "avgpool":
            pooled = F.avg_pool1d(
                importance,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        else:  # maxpool
            pooled = F.max_pool1d(
                importance,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )

        return pooled.squeeze(1)  # (batch, seq_len)

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_scores: Tensor | None = None,
        layer_idx: int = 0,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress KV cache using PyramidKV algorithm.

        Args:
            keys: Key tensor (batch, seq_len, d_model) or (batch, n_heads, seq_len, d_head)
            values: Value tensor with same shape as keys
            attention_scores: Optional pre-computed attention scores
            layer_idx: Current layer index for pyramid allocation

        Returns:
            Tuple of (compressed_keys, compressed_values)
        """
        config = self.pyramid_config

        # Handle 4D input (batch, n_heads, seq_len, d_head)
        is_4d = keys.dim() == 4
        if is_4d:
            batch_size, n_heads, seq_len, d_head = keys.shape
            keys = keys.transpose(1, 2).reshape(batch_size, seq_len, n_heads * d_head)
            values = values.transpose(1, 2).reshape(batch_size, seq_len, n_heads * d_head)
        else:
            batch_size, seq_len, d_model = keys.shape
            n_heads = None

        # Get layer-specific capacity
        capacity = self.compute_layer_capacity(layer_idx)

        # If sequence is shorter than capacity, return as-is
        if seq_len <= capacity:
            if is_4d:
                keys = keys.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
                values = values.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            return keys, values

        # Compute importance scores
        if attention_scores is not None:
            if attention_scores.dim() == 4:
                # (batch, heads, q_len, k_len) -> (batch, k_len)
                importance = attention_scores.mean(dim=1).sum(dim=1)
            elif attention_scores.dim() == 2:
                importance = attention_scores
            else:
                importance = self.compute_attention_scores(keys)
        else:
            importance = self.compute_attention_scores(keys)

        # Apply pooling for smoother selection
        importance = self.apply_pooling(importance)

        # Determine how many tokens to select from middle
        window_size = config.window_size
        num_sink = self.config.num_sink

        # Tokens to select = capacity - window_size - num_sink
        num_to_select = capacity - window_size - num_sink
        num_to_select = max(0, num_to_select)

        if num_to_select == 0:
            # Only keep sink and recent window
            k_out = torch.cat([keys[:, :num_sink], keys[:, -window_size:]], dim=1)
            v_out = torch.cat([values[:, :num_sink], values[:, -window_size:]], dim=1)
        else:
            # Protect sink and recent tokens from selection
            middle_start = num_sink
            middle_end = seq_len - window_size

            if middle_end <= middle_start:
                # Not enough middle tokens, keep all
                k_out = keys
                v_out = values
            else:
                # Get importance of middle tokens
                middle_importance = importance[:, middle_start:middle_end]
                middle_len = middle_end - middle_start

                if num_to_select >= middle_len:
                    # Keep all middle tokens
                    k_out = keys
                    v_out = values
                else:
                    # Select top-k from middle based on importance
                    _, top_indices = middle_importance.topk(
                        min(num_to_select, middle_len), dim=-1
                    )
                    top_indices, _ = top_indices.sort(dim=-1)  # Preserve order

                    # Shift to absolute indices
                    top_indices = top_indices + middle_start

                    # Build complete index set: sink + selected + recent
                    sink_idx = torch.arange(num_sink, device=keys.device)
                    sink_idx = sink_idx.unsqueeze(0).expand(batch_size, -1)

                    recent_idx = torch.arange(
                        seq_len - window_size, seq_len, device=keys.device
                    )
                    recent_idx = recent_idx.unsqueeze(0).expand(batch_size, -1)

                    all_indices = torch.cat([sink_idx, top_indices, recent_idx], dim=-1)
                    all_indices, _ = all_indices.sort(dim=-1)

                    # Gather compressed cache
                    d = keys.shape[-1]
                    k_out = torch.gather(
                        keys, dim=1, index=all_indices.unsqueeze(-1).expand(-1, -1, d)
                    )
                    v_out = torch.gather(
                        values, dim=1, index=all_indices.unsqueeze(-1).expand(-1, -1, d)
                    )

        # Reshape back to 4D if needed
        if is_4d:
            out_len = k_out.shape[1]
            k_out = k_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)
            v_out = v_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)

        return k_out, v_out


def pyramidkv_compress(
    keys: Tensor,
    values: Tensor,
    layer_idx: int = 0,
    num_layers: int = 32,
    max_capacity_prompt: int = 2048,
    window_size: int = 32,
    beta: int = 20,
    num_sink: int = 4,
) -> tuple[Tensor, Tensor]:
    """Standalone PyramidKV function for testing."""
    config = PyramidKVConfig(
        num_sink=num_sink,
        num_recent=window_size,  # Use window_size as num_recent for compatibility
        num_layers=num_layers,
        max_capacity_prompt=max_capacity_prompt,
        window_size=window_size,
        beta=beta,
    )
    method = PyramidKV(config)
    return method.compress(keys, values, layer_idx=layer_idx)
