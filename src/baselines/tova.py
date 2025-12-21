"""
TOVA: Token Omission Via Attention
==================================

Reference: Oren et al. "Transformers are Multi-State RNNs" (arXiv:2401.06104, 2024)

Paper insight: Simple greedy eviction based on current attention scores
works surprisingly well. TOVA evicts the token with the lowest attention
weight at each decoding step.

Algorithm:
1. At each step, compute attention scores for current query
2. Evict the token with the lowest attention score
3. Keep cache at fixed budget size

Key differences from H2O:
- H2O uses cumulative attention, TOVA uses current attention
- TOVA is unbiased (doesn't favor recent tokens implicitly)
- Simple greedy approach with minimal overhead
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .base import CompressionConfig, CompressionMethod


@dataclass
class TOVAConfig(CompressionConfig):
    """TOVA-specific configuration."""

    # Whether to use last query's attention (True) or accumulated (False)
    use_last_attention: bool = True


class TOVA(CompressionMethod):
    """
    TOVA: Token Omission Via Attention.

    Simple greedy eviction policy that removes the currently
    least-attended token once cache budget is exceeded.

    Strategy:
    1. Compute attention scores (or use key norms as proxy)
    2. Evict token with lowest score
    3. Repeat until cache fits budget

    Key insight: Simplicity matters - TOVA's greedy approach
    achieves competitive performance with minimal overhead.
    """

    def __init__(self, config: TOVAConfig):
        super().__init__(config)
        self.tova_config = config

    @property
    def name(self) -> str:
        return "TOVA"

    def _compute_attention_scores(
        self,
        keys: Tensor,
        attention_scores: Tensor | None = None,
    ) -> Tensor:
        """
        Compute attention-based importance scores.

        For post-hoc compression without query, uses key norms.

        Args:
            keys: (batch, seq_len, d_head)
            attention_scores: Optional precomputed attention

        Returns:
            Scores of shape (batch, seq_len)
        """
        if attention_scores is not None:
            if attention_scores.dim() == 2:
                return attention_scores
            elif attention_scores.dim() == 4:
                # Use last row of attention (current query attending to all keys)
                return attention_scores[:, :, -1, :].mean(dim=1)

        # Use key norms as proxy (correlated with attention received)
        return keys.norm(dim=-1)

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_scores: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress KV cache using TOVA eviction.

        Args:
            keys: (batch, seq_len, d_head)
            values: (batch, seq_len, d_head)
            attention_scores: Optional attention weights

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

        num_sink = self.config.num_sink
        num_recent = self.config.num_recent
        budget = self.get_budget(seq_len)

        if seq_len <= budget:
            if is_4d:
                keys = keys.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
                values = values.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            return keys, values

        # Compute attention scores
        scores = self._compute_attention_scores(keys, attention_scores)

        # Protect sink and recent tokens
        protected_scores = scores.clone()
        protected_scores[:, :num_sink] = float("inf")
        if num_recent > 0:
            protected_scores[:, -num_recent:] = float("inf")

        # Find tokens to evict (lowest scores)
        num_to_evict = seq_len - budget
        if num_to_evict <= 0:
            if is_4d:
                keys = keys.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
                values = values.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            return keys, values

        _, evict_indices = protected_scores.topk(num_to_evict, dim=-1, largest=False)

        # Create keep mask
        evict_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=keys.device)
        evict_mask.scatter_(1, evict_indices, True)
        keep_mask = ~evict_mask

        # Gather kept tokens
        keys_out = []
        values_out = []
        for b in range(batch_size):
            keys_out.append(keys[b, keep_mask[b]])
            values_out.append(values[b, keep_mask[b]])

        # Pad to same length
        max_len = max(k.shape[0] for k in keys_out)

        keys_padded = []
        values_padded = []
        for k, v in zip(keys_out, values_out):
            if k.shape[0] < max_len:
                k_pad = torch.zeros(
                    max_len - k.shape[0], k.shape[-1], device=keys.device, dtype=k.dtype
                )
                v_pad = torch.zeros(
                    max_len - v.shape[0], v.shape[-1], device=values.device, dtype=v.dtype
                )
                k = torch.cat([k, k_pad], dim=0)
                v = torch.cat([v, v_pad], dim=0)
            keys_padded.append(k.unsqueeze(0))
            values_padded.append(v.unsqueeze(0))

        keys_out = torch.cat(keys_padded, dim=0)
        values_out = torch.cat(values_padded, dim=0)

        if is_4d:
            out_len = keys_out.shape[1]
            keys_out = keys_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)
            values_out = values_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)

        return keys_out, values_out


def tova_evict(
    keys: Tensor,
    values: Tensor,
    budget: int,
    num_sink: int = 4,
    num_recent: int = 8,
) -> tuple[Tensor, Tensor]:
    """Standalone TOVA function for testing."""
    config = TOVAConfig(num_sink=num_sink, num_recent=num_recent, budget=budget)
    method = TOVA(config)
    return method.compress(keys, values)
