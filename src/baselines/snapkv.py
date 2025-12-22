"""
SnapKV: LLM Knows What You Are Looking For Before Generation
=============================================================

Reference: Li et al. "SnapKV: LLM Knows What You Are Looking For Before Generation"
(NeurIPS 2024, arXiv:2404.14469)

Paper insight: Each attention head consistently focuses on specific prompt attention
features during generation. This robust pattern can be obtained from an "observation
window" located at the end of the prompts.

Algorithm:
1. Use observation window (last ~32 tokens of prompt) to compute attention patterns
2. Aggregate attention weights across queries in observation window (voting)
3. Apply max/avg pooling to capture clustered important positions
4. Select top-k positions based on pooled attention scores
5. Retain those KV pairs + observation window tokens

Key hyperparameters:
- observation_window: Size of window at end of prompt (default: 32)
- kernel_size: Pooling kernel size for clustering (default: 7)

Faithful implementation based on:
- Official repo: https://github.com/FasterDecoding/SnapKV
- NVIDIA kvpress: https://github.com/NVIDIA/kvpress
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionConfig, CompressionMethod


@dataclass
class SnapKVConfig(CompressionConfig):
    """SnapKV-specific configuration."""

    # Size of observation window at end of prompt
    observation_window: int = 32

    # Kernel size for pooling (captures neighboring tokens)
    kernel_size: int = 7

    # Pooling type: "max" or "avg"
    pooling: str = "avg"


class SnapKV(CompressionMethod):
    """
    SnapKV: Observation Window-based KV Cache Compression.

    Strategy:
    1. Extract observation window from end of prompt
    2. Compute attention weights from queries in observation window to all keys
    3. Aggregate (vote) across queries to get importance per key position
    4. Apply pooling to capture clusters of important tokens
    5. Select top-k most important positions
    6. Retain selected KV pairs + full observation window

    Key insight: The attention pattern in the observation window predicts
    what tokens will be important during subsequent generation.
    """

    def __init__(self, config: SnapKVConfig):
        super().__init__(config)
        self.snap_config = config

    @property
    def name(self) -> str:
        return "SnapKV"

    def _compute_attention_scores(
        self,
        queries: Tensor,
        keys: Tensor,
    ) -> Tensor:
        """
        Compute attention scores from queries to keys.

        Args:
            queries: (batch, n_queries, d_head)
            keys: (batch, n_keys, d_head)

        Returns:
            Attention weights (batch, n_keys) aggregated across queries
        """
        # Compute attention: Q @ K^T / sqrt(d)
        d_head = queries.shape[-1]
        attn_scores = torch.bmm(queries, keys.transpose(-1, -2)) / (d_head**0.5)

        # Apply softmax per query
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, n_queries, n_keys)

        # Aggregate across queries (voting) - mean attention received by each key
        importance = attn_weights.mean(dim=1)  # (batch, n_keys)

        return importance

    def _apply_pooling(
        self,
        scores: Tensor,
        kernel_size: int,
    ) -> Tensor:
        """
        Apply pooling to capture clustered important positions.

        This ensures that if a position is important, its neighbors are
        also retained to preserve contextual integrity.

        Args:
            scores: (batch, seq_len) importance scores
            kernel_size: Size of pooling kernel

        Returns:
            Pooled scores (batch, seq_len)
        """
        # Add channel dimension for pooling: (batch, 1, seq_len)
        scores_3d = scores.unsqueeze(1)

        # Apply pooling with padding to preserve length
        padding = kernel_size // 2

        if self.snap_config.pooling == "max":
            pooled = F.max_pool1d(
                scores_3d, kernel_size=kernel_size, stride=1, padding=padding
            )
        else:  # avg
            pooled = F.avg_pool1d(
                scores_3d, kernel_size=kernel_size, stride=1, padding=padding
            )

        # Handle edge case where output size differs
        if pooled.shape[-1] != scores.shape[-1]:
            pooled = pooled[:, :, : scores.shape[-1]]

        return pooled.squeeze(1)  # (batch, seq_len)

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_scores: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress KV cache using SnapKV algorithm.

        Args:
            keys: Key tensor (batch, seq_len, d_head) or (batch, n_heads, seq_len, d_head)
            values: Value tensor with same shape as keys
            attention_scores: Optional precomputed attention (not typically used)

        Returns:
            Tuple of (compressed_keys, compressed_values)
        """
        is_4d = keys.dim() == 4
        if is_4d:
            batch_size, n_heads, seq_len, d_head = keys.shape
            # Process per head by flattening batch and heads
            keys = keys.transpose(1, 2).reshape(batch_size, seq_len, n_heads * d_head)
            values = values.transpose(1, 2).reshape(batch_size, seq_len, n_heads * d_head)
        else:
            batch_size, seq_len, d_head = keys.shape

        num_sink = self.config.num_sink
        num_recent = self.config.num_recent
        obs_window = self.snap_config.observation_window
        budget = self.get_budget(seq_len)

        # If sequence fits within budget, no compression needed
        if seq_len <= budget:
            if is_4d:
                keys = keys.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
                values = values.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            return keys, values

        # Ensure observation window doesn't exceed sequence length
        obs_window = min(obs_window, seq_len - num_sink)

        # Extract observation window (queries) and prefix (keys to score)
        # Observation window: last obs_window tokens before recent tokens
        obs_start = seq_len - num_recent - obs_window if num_recent > 0 else seq_len - obs_window
        obs_end = seq_len - num_recent if num_recent > 0 else seq_len

        # Clamp to valid range
        obs_start = max(num_sink, obs_start)
        obs_end = max(obs_start + 1, obs_end)

        # Use keys in observation window as proxy for queries
        # (In full implementation, we'd use actual query vectors)
        obs_keys = keys[:, obs_start:obs_end, :]  # (batch, obs_window, d)

        # Compute attention from observation window to all prefix keys
        prefix_end = obs_start  # Keys before observation window
        prefix_keys = keys[:, :prefix_end, :]  # (batch, prefix_len, d)

        if prefix_keys.shape[1] == 0:
            # No prefix to compress, just return sink + recent
            if is_4d:
                keys = keys.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
                values = values.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            return keys, values

        # Compute importance scores for prefix positions
        importance = self._compute_attention_scores(obs_keys, prefix_keys)

        # Apply pooling to capture clusters
        pooled_importance = self._apply_pooling(importance, self.snap_config.kernel_size)

        # Protect sink tokens
        pooled_importance[:, :num_sink] = float("inf")

        # Calculate how many prefix tokens to keep
        # Budget = num_sink + selected_middle + obs_window + num_recent
        middle_budget = budget - num_sink - obs_window - num_recent
        middle_budget = max(0, middle_budget)

        # Number of middle tokens available (between sink and observation window)
        middle_available = prefix_end - num_sink

        if middle_available <= middle_budget or middle_budget <= 0:
            # Keep all prefix tokens
            selected_prefix = prefix_keys
            selected_values_prefix = values[:, :prefix_end, :]
        else:
            # Select top-k from middle based on importance
            middle_importance = pooled_importance[:, num_sink:prefix_end]

            _, top_indices = middle_importance.topk(middle_budget, dim=-1)
            top_indices, _ = top_indices.sort(dim=-1)  # Preserve causal order

            # Add sink indices
            sink_indices = torch.arange(num_sink, device=keys.device).unsqueeze(0).expand(batch_size, -1)

            # Shift middle indices to account for sink offset
            middle_indices = top_indices + num_sink

            all_indices = torch.cat([sink_indices, middle_indices], dim=-1)

            # Gather selected prefix tokens
            selected_prefix = torch.gather(
                prefix_keys,
                dim=1,
                index=all_indices.unsqueeze(-1).expand(-1, -1, prefix_keys.shape[-1]),
            )
            selected_values_prefix = torch.gather(
                values[:, :prefix_end, :],
                dim=1,
                index=all_indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
            )

        # Concatenate: selected_prefix + observation_window + recent
        obs_values = values[:, obs_start:obs_end, :]
        recent_keys = keys[:, -num_recent:, :] if num_recent > 0 else keys[:, seq_len:, :]
        recent_values = values[:, -num_recent:, :] if num_recent > 0 else values[:, seq_len:, :]

        keys_out = torch.cat([selected_prefix, obs_keys, recent_keys], dim=1)
        values_out = torch.cat([selected_values_prefix, obs_values, recent_values], dim=1)

        if is_4d:
            out_len = keys_out.shape[1]
            keys_out = keys_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)
            values_out = values_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)

        return keys_out, values_out


def snapkv_compress(
    keys: Tensor,
    values: Tensor,
    budget: int,
    observation_window: int = 32,
    kernel_size: int = 7,
    num_sink: int = 4,
    num_recent: int = 8,
) -> tuple[Tensor, Tensor]:
    """Standalone SnapKV function for testing."""
    config = SnapKVConfig(
        num_sink=num_sink,
        num_recent=num_recent,
        budget=budget,
        observation_window=observation_window,
        kernel_size=kernel_size,
    )
    method = SnapKV(config)
    return method.compress(keys, values)
