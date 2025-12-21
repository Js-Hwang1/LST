"""
H2O: Heavy-Hitter Oracle
========================

Reference: Zhang et al. "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models" (2023)

Paper insight: During generation, a small subset of tokens ("heavy hitters")
receive the majority of attention. By identifying and retaining only these
tokens, we can significantly reduce the KV cache size with minimal quality loss.

Faithful implementation notes:
1. H2O uses ATTENTION SCORES to identify important tokens
2. During decoding, we don't have access to future attention patterns
3. Original paper uses accumulated attention scores from prefill
4. For post-hoc compression, we use key norms as a proxy (correlates with attention)

CRITICAL: This is a SELECTION method, not a compression/merging method.
It evicts tokens entirely, losing their information permanently.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
from torch import Tensor

from .base import CompressionMethod, CompressionConfig, gather_indices


@dataclass
class H2OConfig(CompressionConfig):
    """H2O-specific configuration."""

    # Whether to use attention scores (True) or key norms (False) as proxy
    use_attention_scores: bool = True

    # Whether to keep sink tokens (first few tokens)
    keep_sink: bool = True


class H2O(CompressionMethod):
    """
    H2O: Heavy-Hitter Oracle KV Cache Eviction.

    Keeps the most "important" tokens based on attention scores or key norms.

    Token selection strategy:
    1. Always keep sink tokens (first num_sink tokens)
    2. Always keep recent tokens (last num_recent tokens)
    3. From the middle, select top-k by importance score
    4. Importance = cumulative attention received OR key norm magnitude

    This implementation is faithful to the paper:
    - Uses attention scores when available
    - Falls back to key norms as proxy (correlates with attention)
    - Preserves sink and recent tokens
    """

    def __init__(self, config: H2OConfig):
        super().__init__(config)
        self.h2o_config = config

    @property
    def name(self) -> str:
        return "H2O"

    def compute_importance(
        self,
        keys: Tensor,
        attention_scores: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute importance scores for each token position.

        Args:
            keys: Key tensor (batch, seq_len, d_head) or (batch, n_heads, seq_len, d_head)
            attention_scores: Optional attention weights (batch, n_heads, seq, seq)
                              or precomputed importance (batch, seq)

        Returns:
            Importance scores of shape (batch, seq_len)
        """
        if attention_scores is not None and self.h2o_config.use_attention_scores:
            if attention_scores.dim() == 4:
                # Sum attention received by each position (column sum across queries)
                # attention[..., i, j] = attention from query i to key j
                # We want total attention TO each key position
                importance = attention_scores.mean(dim=1).sum(dim=1)  # (batch, seq)
            elif attention_scores.dim() == 2:
                # Already precomputed importance
                importance = attention_scores
            else:
                raise ValueError(f"Unexpected attention_scores shape: {attention_scores.shape}")
        else:
            # Fallback: use key norms as proxy for importance
            # Empirically, tokens with larger key norms receive more attention
            if keys.dim() == 3:
                importance = keys.norm(dim=-1)  # (batch, seq)
            else:  # dim == 4
                importance = keys.norm(dim=-1).mean(dim=1)  # (batch, seq)

        return importance

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_scores: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compress KV cache using H2O heavy-hitter selection.

        Strategy: Keep sink + recent + top-k important from middle.

        Args:
            keys: Key tensor (batch, seq_len, d_head) or (batch, n_heads, seq_len, d_head)
            values: Value tensor with same shape as keys
            attention_scores: Optional attention weights for importance computation

        Returns:
            Tuple of (compressed_keys, compressed_values)
        """
        is_4d = keys.dim() == 4
        if is_4d:
            batch_size, n_heads, seq_len, d_head = keys.shape
        else:
            batch_size, seq_len, d_head = keys.shape
            n_heads = 1

        num_sink = self.config.num_sink
        num_recent = self.config.num_recent
        budget = self.get_budget(seq_len)

        # If sequence fits within budget, no compression needed
        if seq_len <= budget:
            return keys, values

        # Compute importance scores
        importance = self.compute_importance(keys, attention_scores)

        # Calculate how many middle tokens to select
        middle_budget = budget - num_sink - num_recent
        if middle_budget <= 0:
            # Only keep sink and recent, no middle selection
            if is_4d:
                k_sink = keys[:, :, :num_sink, :]
                v_sink = values[:, :, :num_sink, :]
                k_recent = keys[:, :, -num_recent:, :]
                v_recent = values[:, :, -num_recent:, :]
                return (
                    torch.cat([k_sink, k_recent], dim=2),
                    torch.cat([v_sink, v_recent], dim=2),
                )
            else:
                k_sink = keys[:, :num_sink, :]
                v_sink = values[:, :num_sink, :]
                k_recent = keys[:, -num_recent:, :]
                v_recent = values[:, -num_recent:, :]
                return (
                    torch.cat([k_sink, k_recent], dim=1),
                    torch.cat([v_sink, v_recent], dim=1),
                )

        # Mask out sink and recent positions from selection
        middle_mask = torch.ones(batch_size, seq_len, device=keys.device, dtype=torch.bool)
        middle_mask[:, :num_sink] = False
        middle_mask[:, -num_recent:] = False

        # Set masked positions to -inf for selection
        masked_importance = importance.clone()
        masked_importance[~middle_mask] = float("-inf")

        # Select top-k from middle
        _, middle_indices = masked_importance.topk(middle_budget, dim=-1)
        middle_indices, _ = middle_indices.sort(dim=-1)  # Preserve order

        # Build complete index set
        sink_indices = torch.arange(num_sink, device=keys.device).unsqueeze(0).expand(batch_size, -1)
        recent_indices = torch.arange(seq_len - num_recent, seq_len, device=keys.device).unsqueeze(0).expand(batch_size, -1)
        all_indices = torch.cat([sink_indices, middle_indices, recent_indices], dim=-1)
        all_indices, _ = all_indices.sort(dim=-1)  # Final sort for causal order

        # Gather selected tokens
        keys_compressed = gather_indices(keys, all_indices, dim=1 if not is_4d else 2)
        values_compressed = gather_indices(values, all_indices, dim=1 if not is_4d else 2)

        return keys_compressed, values_compressed


def h2o_evict(
    keys: Tensor,
    values: Tensor,
    budget: int,
    importance: Optional[Tensor] = None,
    num_sink: int = 4,
    num_recent: int = 8,
) -> Tuple[Tensor, Tensor]:
    """
    Standalone H2O eviction function for testing and integration.

    Args:
        keys: Key tensor
        values: Value tensor
        budget: Total number of tokens to retain
        importance: Precomputed importance scores (uses key norms if None)
        num_sink: Number of sink tokens to always keep
        num_recent: Number of recent tokens to always keep

    Returns:
        Tuple of (evicted_keys, evicted_values)
    """
    config = H2OConfig(num_sink=num_sink, num_recent=num_recent, budget=budget)
    method = H2O(config)
    return method.compress(keys, values, attention_scores=importance)
