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
3. Use arithmetic progression for layer budget allocation

Key insight: Since upper layers focus on fewer critical tokens anyway,
we can safely reduce their cache size without losing information.

Faithful implementation based on:
- Paper: https://arxiv.org/abs/2406.02069
- Official repo: https://github.com/Zefan-Cai/KVCache-Factory
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .base import CompressionConfig, CompressionMethod


@dataclass
class PyramidKVConfig(CompressionConfig):
    """PyramidKV-specific configuration."""

    # Total number of layers in the model
    num_layers: int = 32

    # Ratio of budget allocated to lowest vs highest layer
    # Higher = steeper pyramid (more budget in lower layers)
    pyramid_ratio: float = 2.0

    # Whether to use SnapKV-style attention for selection within each layer
    use_attention_selection: bool = True


class PyramidKV(CompressionMethod):
    """
    PyramidKV: Layer-wise Dynamic KV Cache Compression.

    Strategy:
    1. Distribute total budget across layers using pyramid allocation
       - Lower layers get more budget (attention is scattered)
       - Upper layers get less budget (attention focuses on sinks)
    2. Within each layer, select most important tokens based on:
       - Attention scores if available
       - Key norms as proxy otherwise

    The pyramid allocation uses arithmetic progression:
    budget[layer] = base + (num_layers - 1 - layer) * step

    This ensures:
    - Layer 0 gets the most budget
    - Layer num_layers-1 gets the least budget
    - Total budget matches target compression ratio
    """

    def __init__(self, config: PyramidKVConfig):
        super().__init__(config)
        self.pyramid_config = config

    @property
    def name(self) -> str:
        return "PyramidKV"

    def compute_layer_budgets(
        self,
        total_budget: int,
        num_layers: int,
        seq_len: int,
    ) -> list[int]:
        """
        Compute per-layer KV cache budgets using pyramid allocation.

        Uses arithmetic progression where:
        - Layer 0 (lowest) gets max_budget
        - Layer num_layers-1 (highest) gets min_budget
        - Sum of all budgets = total_budget

        Args:
            total_budget: Total number of KV pairs to retain across all layers
            num_layers: Number of transformer layers
            seq_len: Original sequence length

        Returns:
            List of budgets for each layer
        """
        ratio = self.pyramid_config.pyramid_ratio

        # For arithmetic progression: a_i = a_0 - i * d
        # where a_0 is the largest (layer 0), a_{n-1} is smallest
        # Sum = n * (a_0 + a_{n-1}) / 2 = total_budget
        # a_0 / a_{n-1} = ratio

        # Let a_{n-1} = min_budget, a_0 = ratio * min_budget
        # Sum = n * min_budget * (1 + ratio) / 2 = total_budget
        # min_budget = 2 * total_budget / (n * (1 + ratio))

        min_budget = 2 * total_budget / (num_layers * (1 + ratio))
        max_budget = ratio * min_budget

        # Step size for arithmetic progression
        if num_layers > 1:
            step = (max_budget - min_budget) / (num_layers - 1)
        else:
            step = 0

        # Compute per-layer budgets
        budgets = []
        for layer_idx in range(num_layers):
            budget = max_budget - layer_idx * step
            # Ensure minimum of num_sink + num_recent
            min_required = self.config.num_sink + self.config.num_recent + 1
            budget = max(min_required, int(budget))
            # Don't exceed sequence length
            budget = min(budget, seq_len)
            budgets.append(budget)

        return budgets

    def compress_single_layer(
        self,
        keys: Tensor,
        values: Tensor,
        budget: int,
        attention_scores: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress a single layer's KV cache to the given budget.

        Args:
            keys: (batch, seq_len, d_head)
            values: (batch, seq_len, d_head)
            budget: Number of tokens to retain
            attention_scores: Optional attention weights

        Returns:
            Compressed (keys, values)
        """
        batch_size, seq_len, d_head = keys.shape

        if seq_len <= budget:
            return keys, values

        num_sink = self.config.num_sink
        num_recent = self.config.num_recent

        # Compute importance scores
        if attention_scores is not None and attention_scores.dim() >= 2:
            if attention_scores.dim() == 4:
                # (batch, heads, seq, seq) -> aggregate
                importance = attention_scores.mean(dim=1).sum(dim=1)
            elif attention_scores.dim() == 2:
                importance = attention_scores
            else:
                importance = keys.norm(dim=-1)
        else:
            # Use key norms as proxy
            importance = keys.norm(dim=-1)

        # Protect sink and recent
        protected_importance = importance.clone()
        protected_importance[:, :num_sink] = float("inf")
        if num_recent > 0:
            protected_importance[:, -num_recent:] = float("inf")

        # Select top tokens from middle
        middle_budget = budget - num_sink - num_recent
        middle_budget = max(0, middle_budget)

        if middle_budget <= 0:
            # Only keep sink and recent
            k_sink = keys[:, :num_sink, :]
            v_sink = values[:, :num_sink, :]
            k_recent = keys[:, -num_recent:, :] if num_recent > 0 else keys[:, seq_len:, :]
            v_recent = values[:, -num_recent:, :] if num_recent > 0 else values[:, seq_len:, :]
            return torch.cat([k_sink, k_recent], dim=1), torch.cat([v_sink, v_recent], dim=1)

        # Get middle importance (exclude sink and recent)
        middle_start = num_sink
        middle_end = seq_len - num_recent if num_recent > 0 else seq_len
        middle_importance = importance[:, middle_start:middle_end]

        middle_len = middle_end - middle_start
        if middle_len <= middle_budget:
            # Keep all middle tokens
            return keys, values

        # Select top-k from middle
        _, top_indices = middle_importance.topk(middle_budget, dim=-1)
        top_indices, _ = top_indices.sort(dim=-1)  # Preserve order

        # Shift to absolute indices
        top_indices = top_indices + middle_start

        # Build index set
        sink_indices = torch.arange(num_sink, device=keys.device).unsqueeze(0).expand(batch_size, -1)
        recent_indices = (
            torch.arange(seq_len - num_recent, seq_len, device=keys.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        ) if num_recent > 0 else torch.empty(batch_size, 0, dtype=torch.long, device=keys.device)

        all_indices = torch.cat([sink_indices, top_indices, recent_indices], dim=-1)
        all_indices, _ = all_indices.sort(dim=-1)

        # Gather
        keys_out = torch.gather(
            keys, dim=1, index=all_indices.unsqueeze(-1).expand(-1, -1, d_head)
        )
        values_out = torch.gather(
            values, dim=1, index=all_indices.unsqueeze(-1).expand(-1, -1, d_head)
        )

        return keys_out, values_out

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

        Note: For full PyramidKV, this should be called per-layer with layer_idx.
        For post-hoc compression of full cache, we apply average budget.

        Args:
            keys: Key tensor (batch, seq_len, d_head) or (batch, n_heads, seq_len, d_head)
            values: Value tensor with same shape as keys
            attention_scores: Optional attention weights
            layer_idx: Current layer index (for per-layer budget allocation)

        Returns:
            Tuple of (compressed_keys, compressed_values)
        """
        is_4d = keys.dim() == 4
        if is_4d:
            batch_size, n_heads, seq_len, d_head = keys.shape
            keys = keys.transpose(1, 2).reshape(batch_size, seq_len, n_heads * d_head)
            values = values.transpose(1, 2).reshape(batch_size, seq_len, n_heads * d_head)
        else:
            batch_size, seq_len, d_head = keys.shape

        # Get base budget
        base_budget = self.get_budget(seq_len)

        if seq_len <= base_budget:
            if is_4d:
                keys = keys.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
                values = values.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            return keys, values

        # Compute layer-specific budget if layer_idx is meaningful
        num_layers = self.pyramid_config.num_layers
        layer_budgets = self.compute_layer_budgets(
            total_budget=base_budget * num_layers,
            num_layers=num_layers,
            seq_len=seq_len,
        )

        # Use budget for this layer
        if 0 <= layer_idx < len(layer_budgets):
            budget = layer_budgets[layer_idx]
        else:
            budget = base_budget

        # Compress this layer
        keys_out, values_out = self.compress_single_layer(
            keys, values, budget, attention_scores
        )

        if is_4d:
            out_len = keys_out.shape[1]
            keys_out = keys_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)
            values_out = values_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)

        return keys_out, values_out


def pyramidkv_compress(
    keys: Tensor,
    values: Tensor,
    budget: int,
    layer_idx: int = 0,
    num_layers: int = 32,
    pyramid_ratio: float = 2.0,
    num_sink: int = 4,
    num_recent: int = 8,
) -> tuple[Tensor, Tensor]:
    """Standalone PyramidKV function for testing."""
    config = PyramidKVConfig(
        num_sink=num_sink,
        num_recent=num_recent,
        budget=budget,
        num_layers=num_layers,
        pyramid_ratio=pyramid_ratio,
    )
    method = PyramidKV(config)
    return method.compress(keys, values, layer_idx=layer_idx)
