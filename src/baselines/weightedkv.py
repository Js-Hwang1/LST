"""
WeightedKV: Attention Scores Weighted Key-Value Cache Merging
==============================================================

Reference: Yuan et al. "WeightedKV: Attention Scores Weighted Key-Value Cache
Merging for Large Language Models" (arXiv:2503.01330, ICASSP 2025)

Paper insight: Values don't exhibit a strong low-rank property like keys do,
meaning information is distributed more evenly across values. Therefore,
evicting both keys AND values loses crucial information.

Solution: Discard keys of less important tokens, but MERGE their values into
neighboring tokens using attention-weighted convex combination.

Algorithm:
1. Rank tokens by accumulated attention scores
2. For tokens to evict: discard their keys
3. Merge their values into neighbors: v_merged = (ā[j]·v[j] + ā[j+1]·v[j+1]) / (ā[j] + ā[j+1])
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .base import CompressionConfig, CompressionMethod


@dataclass
class WeightedKVConfig(CompressionConfig):
    """WeightedKV-specific configuration."""

    # Whether to merge values (True) or just evict (False)
    merge_values: bool = True


class WeightedKV(CompressionMethod):
    """
    WeightedKV: Asymmetric Key Eviction with Value Merging.

    Strategy:
    1. Compute importance scores (use key norms as proxy for attention)
    2. Identify least important tokens to evict
    3. Discard their keys entirely
    4. Merge their values into neighboring tokens using weighted average

    Key insight: Retained keys act as "anchors" guiding generation,
    while merged values provide rich contextual backdrop.
    """

    def __init__(self, config: WeightedKVConfig):
        super().__init__(config)
        self.weighted_config = config

    @property
    def name(self) -> str:
        return "WeightedKV"

    def _compute_importance(
        self,
        keys: Tensor,
        attention_scores: Tensor | None = None,
    ) -> Tensor:
        """
        Compute importance scores for each token.

        Uses accumulated attention or key norms as proxy.

        Args:
            keys: (batch, seq_len, d_head)
            attention_scores: Optional precomputed attention scores

        Returns:
            Importance scores of shape (batch, seq_len)
        """
        if attention_scores is not None:
            if attention_scores.dim() == 2:
                return attention_scores
            elif attention_scores.dim() == 4:
                # Sum attention received by each key position
                return attention_scores.mean(dim=1).sum(dim=1)

        # Use key norms as proxy for importance
        return keys.norm(dim=-1)

    def _merge_values_weighted(
        self,
        values: Tensor,
        importance: Tensor,
        evict_mask: Tensor,
    ) -> Tensor:
        """
        Merge values of evicted tokens into their neighbors.

        For each evicted token j, merge its value into neighbor j+1:
        v_merged[j+1] = (imp[j]·v[j] + imp[j+1]·v[j+1]) / (imp[j] + imp[j+1])

        Args:
            values: (batch, seq_len, d_head)
            importance: (batch, seq_len) importance scores
            evict_mask: (batch, seq_len) True for tokens to evict

        Returns:
            Merged values with evicted positions removed
        """
        batch_size, seq_len, d_head = values.shape
        device = values.device

        merged_values = []

        for b in range(batch_size):
            evict_idx = evict_mask[b].nonzero(as_tuple=True)[0]
            keep_idx = (~evict_mask[b]).nonzero(as_tuple=True)[0]

            if len(evict_idx) == 0:
                merged_values.append(values[b])
                continue

            v_out = values[b].clone()
            imp = importance[b]

            # For each evicted token, merge into next kept token
            for idx in evict_idx:
                idx = idx.item()
                # Find next kept token
                next_kept = keep_idx[keep_idx > idx]
                if len(next_kept) == 0:
                    # Find previous kept token if no next
                    prev_kept = keep_idx[keep_idx < idx]
                    if len(prev_kept) == 0:
                        continue
                    neighbor = prev_kept[-1].item()
                else:
                    neighbor = next_kept[0].item()

                # Weighted merge
                w_evict = imp[idx]
                w_neighbor = imp[neighbor]
                total_w = w_evict + w_neighbor + 1e-8

                v_out[neighbor] = (
                    w_evict * values[b, idx] + w_neighbor * values[b, neighbor]
                ) / total_w

            # Keep only non-evicted positions
            merged_values.append(v_out[keep_idx])

        # Pad to same length
        max_len = max(v.shape[0] for v in merged_values)
        padded = []
        for v in merged_values:
            if v.shape[0] < max_len:
                pad = torch.zeros(max_len - v.shape[0], d_head, device=device, dtype=v.dtype)
                v = torch.cat([v, pad], dim=0)
            padded.append(v.unsqueeze(0))

        return torch.cat(padded, dim=0)

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_scores: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress KV cache using WeightedKV algorithm.

        Args:
            keys: (batch, seq_len, d_head)
            values: (batch, seq_len, d_head)
            attention_scores: Optional attention weights for importance

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

        # Compute importance
        importance = self._compute_importance(keys, attention_scores)

        # Mask sink and recent tokens (always keep)
        keep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=keys.device)
        keep_mask[:, :num_sink] = True
        keep_mask[:, -num_recent:] = True

        # Find tokens to evict from middle
        middle_importance = importance.clone()
        middle_importance[:, :num_sink] = float("inf")
        middle_importance[:, -num_recent:] = float("inf")

        num_to_evict = seq_len - budget
        _, evict_indices = middle_importance.topk(num_to_evict, dim=-1, largest=False)

        evict_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=keys.device)
        evict_mask.scatter_(1, evict_indices, True)

        # Compress keys (simple eviction)
        keys_out = []
        for b in range(batch_size):
            k_keep = keys[b, ~evict_mask[b]]
            keys_out.append(k_keep)

        # Pad keys to same length
        max_len = max(k.shape[0] for k in keys_out)
        keys_padded = []
        for k in keys_out:
            if k.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - k.shape[0], k.shape[-1], device=keys.device, dtype=k.dtype
                )
                k = torch.cat([k, pad], dim=0)
            keys_padded.append(k.unsqueeze(0))
        keys_out = torch.cat(keys_padded, dim=0)

        # Compress values (with merging if enabled)
        if self.weighted_config.merge_values:
            values_out = self._merge_values_weighted(values, importance, evict_mask)
        else:
            # Simple eviction
            values_out = []
            for b in range(batch_size):
                v_keep = values[b, ~evict_mask[b]]
                values_out.append(v_keep)

            max_len = max(v.shape[0] for v in values_out)
            values_padded = []
            for v in values_out:
                if v.shape[0] < max_len:
                    pad = torch.zeros(
                        max_len - v.shape[0], v.shape[-1], device=values.device, dtype=v.dtype
                    )
                    v = torch.cat([v, pad], dim=0)
                values_padded.append(v.unsqueeze(0))
            values_out = torch.cat(values_padded, dim=0)

        if is_4d:
            out_len = keys_out.shape[1]
            keys_out = keys_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)
            values_out = values_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)

        return keys_out, values_out


def weightedkv_compress(
    keys: Tensor,
    values: Tensor,
    budget: int,
    num_sink: int = 4,
    num_recent: int = 8,
    merge_values: bool = True,
) -> tuple[Tensor, Tensor]:
    """Standalone WeightedKV function for testing."""
    config = WeightedKVConfig(
        num_sink=num_sink,
        num_recent=num_recent,
        budget=budget,
        merge_values=merge_values,
    )
    method = WeightedKV(config)
    return method.compress(keys, values)
