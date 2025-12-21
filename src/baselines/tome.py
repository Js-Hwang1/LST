"""
ToMe: Token Merging
===================

Reference: Bolya et al. "Token Merging: Your ViT But Faster" (2023)

Paper insight: Merge similar tokens based on key similarity. Unlike eviction
methods (H2O, StreamingLLM), ToMe preserves information by averaging similar
tokens instead of discarding them.

Original context: Vision Transformers (ViT), but adaptable to LLM KV caches.

Faithful implementation notes:
1. Compute pairwise similarity between token keys
2. Partition tokens into source (to merge) and destination (to keep)
3. Merge each source token with its most similar destination token
4. Use weighted average for merging (not just replacement)

CRITICAL: ToMe is a COMPRESSION method that preserves information,
unlike eviction methods that lose information entirely.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionConfig, CompressionMethod


@dataclass
class ToMeConfig(CompressionConfig):
    """ToMe-specific configuration."""

    # Number of tokens to merge per step
    r: int = 4  # Merge ratio

    # Similarity metric: "cosine" or "dot"
    similarity: str = "cosine"

    # Whether to use bipartite matching (original ToMe) or greedy
    bipartite: bool = True


class ToMe(CompressionMethod):
    """
    ToMe: Token Merging for KV Cache Compression.

    Strategy:
    1. Compute pairwise key similarities
    2. Partition into source (odd indices) and destination (even indices)
    3. For each source, find most similar destination
    4. Merge by weighted averaging

    Key difference from eviction:
    - Information from merged tokens is PRESERVED (averaged)
    - Better for long-range dependencies
    - Higher computational cost than simple eviction
    """

    def __init__(self, config: ToMeConfig):
        super().__init__(config)
        self.tome_config = config

    @property
    def name(self) -> str:
        return "ToMe"

    def compute_similarity(
        self,
        src_keys: Tensor,
        dst_keys: Tensor,
    ) -> Tensor:
        """
        Compute similarity matrix between source and destination keys.

        Args:
            src_keys: (batch, n_src, d_head)
            dst_keys: (batch, n_dst, d_head)

        Returns:
            Similarity matrix of shape (batch, n_src, n_dst)
        """
        if self.tome_config.similarity == "cosine":
            src_norm = F.normalize(src_keys, dim=-1)
            dst_norm = F.normalize(dst_keys, dim=-1)
            return torch.bmm(src_norm, dst_norm.transpose(-1, -2))
        else:  # dot product
            return torch.bmm(src_keys, dst_keys.transpose(-1, -2))

    def bipartite_matching(
        self,
        similarity: Tensor,
        r: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Bipartite matching to find optimal merge pairs.

        Following original ToMe paper:
        1. Partition tokens into source (odd) and destination (even)
        2. For each source, find most similar destination
        3. Merge top-r most similar pairs

        Args:
            similarity: (batch, n_src, n_dst) similarity matrix
            r: Number of merges to perform

        Returns:
            Tuple of (src_indices, dst_indices) for top-r merges
        """
        batch_size, n_src, n_dst = similarity.shape

        # Find best destination for each source
        max_sim, dst_indices = similarity.max(dim=-1)  # (batch, n_src)

        # Select top-r source tokens to merge
        _, top_src = max_sim.topk(min(r, n_src), dim=-1)  # (batch, r)

        # Get corresponding destination indices
        batch_idx = torch.arange(batch_size, device=similarity.device).unsqueeze(-1)
        top_dst = dst_indices[batch_idx, top_src]  # (batch, r)

        return top_src, top_dst

    def merge_tokens(
        self,
        keys: Tensor,
        values: Tensor,
        src_indices: Tensor,
        dst_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Merge source tokens into destination tokens.

        Args:
            keys: (batch, seq_len, d_head)
            values: (batch, seq_len, d_head)
            src_indices: (batch, r) indices of source tokens
            dst_indices: (batch, r) indices of destination tokens

        Returns:
            Tuple of (merged_keys, merged_values)
        """
        batch_size, seq_len, d_head = keys.shape
        r = src_indices.shape[-1]

        # Create output tensors
        keys_out = keys.clone()
        values_out = values.clone()

        # Merge: average source into destination
        # Simple averaging (could use weighted average with similarity)
        for b in range(batch_size):
            for i in range(r):
                s_idx = src_indices[b, i]
                d_idx = dst_indices[b, i]
                # Average the tokens
                keys_out[b, d_idx] = (keys_out[b, d_idx] + keys[b, s_idx]) / 2
                values_out[b, d_idx] = (values_out[b, d_idx] + values[b, s_idx]) / 2

        # Remove source tokens (they've been merged into destinations)
        # Create mask for tokens to keep
        keep_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=keys.device)
        keep_mask.scatter_(1, src_indices, False)

        # Gather kept tokens
        kept_indices = []
        for b in range(batch_size):
            kept_idx = keep_mask[b].nonzero(as_tuple=True)[0]
            kept_indices.append(kept_idx)

        # Stack and gather (assuming same number of kept tokens per batch)
        # This is a simplification - real implementation would handle variable lengths
        min_kept = min(len(idx) for idx in kept_indices)
        kept_indices = torch.stack([idx[:min_kept] for idx in kept_indices])

        keys_merged = keys_out.gather(1, kept_indices.unsqueeze(-1).expand(-1, -1, d_head))
        values_merged = values_out.gather(1, kept_indices.unsqueeze(-1).expand(-1, -1, d_head))

        return keys_merged, values_merged

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_scores: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress KV cache using Token Merging.

        Strategy:
        1. Partition tokens: even indices as destinations, odd as sources
        2. Compute similarity between sources and destinations
        3. Merge top-r most similar pairs
        4. Repeat until budget is met

        Args:
            keys: Key tensor (batch, seq_len, d_head)
            values: Value tensor with same shape as keys
            attention_scores: Ignored (ToMe uses key similarity)

        Returns:
            Tuple of (merged_keys, merged_values)
        """
        is_4d = keys.dim() == 4
        if is_4d:
            # Handle 4D case by flattening heads
            batch_size, n_heads, seq_len, d_head = keys.shape
            keys = keys.transpose(1, 2).reshape(batch_size, seq_len, n_heads * d_head)
            values = values.transpose(1, 2).reshape(batch_size, seq_len, n_heads * d_head)
        else:
            batch_size, seq_len, d_head = keys.shape

        budget = self.get_budget(seq_len)

        # If sequence fits within budget, no merging needed
        if seq_len <= budget:
            if is_4d:
                keys = keys.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
                values = values.view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            return keys, values

        # Protect sink and recent tokens
        num_sink = self.config.num_sink
        num_recent = self.config.num_recent

        # Extract sink, middle, and recent
        k_sink = keys[:, :num_sink, :]
        v_sink = values[:, :num_sink, :]
        k_recent = keys[:, -num_recent:, :] if num_recent > 0 else keys[:, seq_len:, :]
        v_recent = values[:, -num_recent:, :] if num_recent > 0 else values[:, seq_len:, :]

        # Only merge middle tokens
        middle_end = seq_len - num_recent if num_recent > 0 else seq_len
        k_middle = keys[:, num_sink:middle_end, :]
        v_middle = values[:, num_sink:middle_end, :]

        middle_len = k_middle.shape[1]
        target_middle = budget - num_sink - num_recent

        if middle_len <= target_middle or target_middle <= 0:
            # No merging needed in middle
            keys_out = torch.cat([k_sink, k_middle, k_recent], dim=1)
            values_out = torch.cat([v_sink, v_middle, v_recent], dim=1)
        else:
            # Iteratively merge until we reach target
            k_merged = k_middle
            v_merged = v_middle

            while k_merged.shape[1] > target_middle:
                curr_len = k_merged.shape[1]
                r = min(self.tome_config.r, curr_len - target_middle)
                if r <= 0:
                    break

                # Partition: even as dst, odd as src
                n_src = curr_len // 2

                if n_src == 0:
                    break

                dst_keys = k_merged[:, ::2, :]  # Even indices
                src_keys = k_merged[:, 1::2, :]  # Odd indices

                # Compute similarity
                similarity = self.compute_similarity(src_keys, dst_keys)

                # Find best matches
                src_idx, dst_idx = self.bipartite_matching(similarity, r)

                # Build full-sequence indices
                full_src_idx = 1 + 2 * src_idx  # Map back to odd indices
                full_dst_idx = 2 * dst_idx  # Map back to even indices

                # Merge
                k_merged, v_merged = self.merge_tokens(
                    k_merged, v_merged, full_src_idx, full_dst_idx
                )

            keys_out = torch.cat([k_sink, k_merged, k_recent], dim=1)
            values_out = torch.cat([v_sink, v_merged, v_recent], dim=1)

        if is_4d:
            out_len = keys_out.shape[1]
            keys_out = keys_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)
            values_out = values_out.view(batch_size, out_len, n_heads, -1).transpose(1, 2)

        return keys_out, values_out


def tome_merge(
    keys: Tensor,
    values: Tensor,
    budget: int,
    r: int = 4,
    num_sink: int = 4,
    num_recent: int = 8,
) -> tuple[Tensor, Tensor]:
    """
    Standalone ToMe merge function for testing and integration.

    Args:
        keys: Key tensor
        values: Value tensor
        budget: Total number of tokens to retain
        r: Number of tokens to merge per iteration
        num_sink: Number of sink tokens to protect
        num_recent: Number of recent tokens to protect

    Returns:
        Tuple of (merged_keys, merged_values)
    """
    config = ToMeConfig(num_sink=num_sink, num_recent=num_recent, budget=budget, r=r)
    method = ToMe(config)
    return method.compress(keys, values)
