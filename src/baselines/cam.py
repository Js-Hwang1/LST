"""
CaM: Cache Merging for Memory-efficient LLMs Inference
=======================================================

Reference: Zhang et al. "CaM: Cache Merging for Memory-efficient LLMs Inference"
(ICML 2024)

Paper insight: KV cache eviction invariably leads to output perturbation,
regardless of token choice. This perturbation escalates with compression ratio.

Solution: Instead of pure eviction, MERGE to-be-evicted caches into remaining
ones using a sampling strategy governed by attention score prominence.

Algorithm:
1. Select tokens to evict based on attention scores
2. For each evicted token, merge its value into remaining tokens
3. Use attention scores to weight the merging contribution
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionMethod, CompressionConfig


@dataclass
class CaMConfig(CompressionConfig):
    """CaM-specific configuration."""

    # Merging strategy: "nearest" or "weighted"
    merge_strategy: str = "weighted"


class CaM(CompressionMethod):
    """
    CaM: Cache Merging for Memory-efficient LLMs Inference.

    Strategy:
    1. Identify tokens to evict based on attention importance
    2. Merge evicted value states into remaining value states
    3. Weight merging by attention score prominence

    Key insight: Merging preserves more information than pure eviction,
    reducing output perturbation especially at high compression ratios.
    """

    def __init__(self, config: CaMConfig):
        super().__init__(config)
        self.cam_config = config

    @property
    def name(self) -> str:
        return "CaM"

    def _compute_importance(
        self,
        keys: Tensor,
        attention_scores: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute importance scores for token selection."""
        if attention_scores is not None and attention_scores.dim() == 2:
            return attention_scores
        # Use key norms as proxy
        return keys.norm(dim=-1)

    def _merge_into_remaining(
        self,
        values: Tensor,
        importance: Tensor,
        evict_mask: Tensor,
        keep_mask: Tensor,
    ) -> Tensor:
        """
        Merge evicted values into remaining values.

        Each evicted value is distributed to kept values proportionally
        to the kept values' importance scores.

        Args:
            values: (batch, seq_len, d_head)
            importance: (batch, seq_len)
            evict_mask: (batch, seq_len) True for tokens to evict
            keep_mask: (batch, seq_len) True for tokens to keep

        Returns:
            Merged values at kept positions
        """
        batch_size, seq_len, d_head = values.shape
        device = values.device

        merged_values = []

        for b in range(batch_size):
            keep_idx = keep_mask[b].nonzero(as_tuple=True)[0]
            evict_idx = evict_mask[b].nonzero(as_tuple=True)[0]

            if len(evict_idx) == 0 or len(keep_idx) == 0:
                merged_values.append(values[b, keep_idx])
                continue

            # Start with kept values
            v_out = values[b, keep_idx].clone()  # (num_keep, d_head)

            # Importance of kept tokens for weighting
            keep_importance = importance[b, keep_idx]  # (num_keep,)
            keep_weights = F.softmax(keep_importance, dim=0)  # Normalized

            # For each evicted token, distribute its value to kept tokens
            for idx in evict_idx:
                v_evict = values[b, idx]  # (d_head,)
                evict_imp = importance[b, idx]

                # Weight by relative importance
                contribution = evict_imp * v_evict.unsqueeze(0) * keep_weights.unsqueeze(-1)
                v_out = v_out + contribution

            merged_values.append(v_out)

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
        attention_scores: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compress KV cache using CaM algorithm.

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

        # Compute importance
        importance = self._compute_importance(keys, attention_scores)

        # Always keep sink and recent
        keep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=keys.device)
        keep_mask[:, :num_sink] = True
        if num_recent > 0:
            keep_mask[:, -num_recent:] = True

        # Find tokens to evict from middle
        middle_importance = importance.clone()
        middle_importance[:, :num_sink] = float("inf")
        if num_recent > 0:
            middle_importance[:, -num_recent:] = float("inf")

        num_to_evict = seq_len - budget
        if num_to_evict <= 0:
            if is_4d:
                keys = keys.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
                values = values.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            return keys, values

        _, evict_indices = middle_importance.topk(num_to_evict, dim=-1, largest=False)

        evict_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=keys.device)
        evict_mask.scatter_(1, evict_indices, True)

        # Update keep_mask
        keep_mask = ~evict_mask

        # Evict keys (no merging for keys)
        keys_out = []
        for b in range(batch_size):
            keys_out.append(keys[b, keep_mask[b]])

        max_len = max(k.shape[0] for k in keys_out)
        keys_padded = []
        for k in keys_out:
            if k.shape[0] < max_len:
                pad = torch.zeros(max_len - k.shape[0], k.shape[-1], device=keys.device, dtype=k.dtype)
                k = torch.cat([k, pad], dim=0)
            keys_padded.append(k.unsqueeze(0))
        keys_out = torch.cat(keys_padded, dim=0)

        # Merge values
        values_out = self._merge_into_remaining(values, importance, evict_mask, keep_mask)

        if is_4d:
            out_len = keys_out.shape[1]
            keys_out = keys_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)
            values_out = values_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)

        return keys_out, values_out


def cam_compress(
    keys: Tensor,
    values: Tensor,
    budget: int,
    num_sink: int = 4,
    num_recent: int = 8,
) -> Tuple[Tensor, Tensor]:
    """Standalone CaM function for testing."""
    config = CaMConfig(num_sink=num_sink, num_recent=num_recent, budget=budget)
    method = CaM(config)
    return method.compress(keys, values)
