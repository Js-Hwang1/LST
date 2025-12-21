"""
KVMerger: Adaptive KV Cache Merging for LLMs on Long-Context Tasks
===================================================================

Reference: Wang et al. "Model Tells You Where to Merge: Adaptive KV Cache
Merging for LLMs on Long-Context Tasks" (arXiv:2407.08454, 2024)

Paper insight: Key states exhibit high similarity at the token level within
a single sequence. By identifying clusters of similar keys and merging them
using Gaussian kernel weighting, we can compress the cache while preserving
information.

Algorithm:
1. Identify merging sets: Group adjacent tokens with cosine similarity > threshold
2. For each merging set, select pivot (highest attention score)
3. Merge using Gaussian kernel: g_i = exp(-||k_p - k_i||² / (2σ²))
4. Merged K, V = weighted sum with normalized Gaussian weights
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionMethod, CompressionConfig


@dataclass
class KVMergerConfig(CompressionConfig):
    """KVMerger-specific configuration."""

    # Similarity threshold for merging (paper uses 0.75)
    similarity_threshold: float = 0.75

    # Minimum cluster size for merging
    min_cluster_size: int = 2

    # Target compression ratio (final_len / original_len)
    target_ratio: float = 0.5


class KVMerger(CompressionMethod):
    """
    KVMerger: Gaussian Kernel Weighted KV Cache Merging.

    Strategy:
    1. Compute pairwise cosine similarity between adjacent keys
    2. Identify merging sets (consecutive tokens with similarity > threshold)
    3. For each set, compute Gaussian kernel weights from pivot
    4. Merge K and V using normalized Gaussian weights

    Key insight: Adjacent tokens often have highly similar key states,
    so merging them preserves most of the attention information.
    """

    def __init__(self, config: KVMergerConfig):
        super().__init__(config)
        self.merger_config = config

    @property
    def name(self) -> str:
        return "KVMerger"

    def _compute_adjacent_similarity(self, keys: Tensor) -> Tensor:
        """
        Compute cosine similarity between adjacent key tokens.

        Args:
            keys: (batch, seq_len, d_head)

        Returns:
            Similarity scores of shape (batch, seq_len - 1)
        """
        keys_norm = F.normalize(keys, dim=-1)
        # Similarity between token i and token i+1
        sim = (keys_norm[:, :-1, :] * keys_norm[:, 1:, :]).sum(dim=-1)
        return sim

    def _identify_merging_sets(
        self,
        similarity: Tensor,
        threshold: float,
    ) -> List[List[Tuple[int, int]]]:
        """
        Identify contiguous regions where similarity exceeds threshold.

        Args:
            similarity: (batch, seq_len - 1) adjacent similarities
            threshold: Similarity threshold for merging

        Returns:
            List of merging sets per batch, each set is (start, end) indices
        """
        batch_size, seq_len_minus_1 = similarity.shape
        all_sets = []

        for b in range(batch_size):
            sets = []
            sim_b = similarity[b].cpu().numpy()

            i = 0
            while i < len(sim_b):
                if sim_b[i] >= threshold:
                    # Start of a merging set
                    start = i
                    while i < len(sim_b) and sim_b[i] >= threshold:
                        i += 1
                    end = i + 1  # Include the last token
                    if end - start >= self.merger_config.min_cluster_size:
                        sets.append((start, end))
                else:
                    i += 1

            all_sets.append(sets)

        return all_sets

    def _gaussian_kernel_merge(
        self,
        keys: Tensor,
        values: Tensor,
        start: int,
        end: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Merge a set of tokens using Gaussian kernel weighting.

        The pivot is the center token. Weights are computed as:
        g_i = exp(-||k_pivot - k_i||² / (2σ²))

        Args:
            keys: (batch, seq_len, d_head)
            values: (batch, seq_len, d_head)
            start, end: Indices defining the merging set

        Returns:
            Merged (key, value), each of shape (batch, d_head)
        """
        k_set = keys[:, start:end, :]  # (batch, set_len, d_head)
        v_set = values[:, start:end, :]

        set_len = end - start
        pivot_idx = set_len // 2  # Use center as pivot

        k_pivot = k_set[:, pivot_idx:pivot_idx + 1, :]  # (batch, 1, d_head)

        # Compute squared distances from pivot
        diff = k_set - k_pivot  # (batch, set_len, d_head)
        sq_dist = (diff ** 2).sum(dim=-1)  # (batch, set_len)

        # Compute sigma as mean distance
        sigma = sq_dist.mean(dim=-1, keepdim=True).sqrt() + 1e-8  # (batch, 1)

        # Gaussian kernel weights
        weights = torch.exp(-sq_dist / (2 * sigma ** 2 + 1e-8))  # (batch, set_len)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize

        # Weighted merge
        weights = weights.unsqueeze(-1)  # (batch, set_len, 1)
        k_merged = (weights * k_set).sum(dim=1)  # (batch, d_head)
        v_merged = (weights * v_set).sum(dim=1)  # (batch, d_head)

        return k_merged, v_merged

    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_scores: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compress KV cache using KVMerger algorithm.

        Args:
            keys: (batch, seq_len, d_head)
            values: (batch, seq_len, d_head)
            attention_scores: Optional attention weights (not used in basic version)

        Returns:
            Tuple of (compressed_keys, compressed_values)
        """
        is_4d = keys.dim() == 4
        if is_4d:
            batch_size, n_heads, seq_len, d_head = keys.shape
            # Process per head by flattening
            keys = keys.transpose(1, 2).reshape(batch_size * n_heads, seq_len, d_head)
            values = values.transpose(1, 2).reshape(batch_size * n_heads, seq_len, d_head)
        else:
            batch_size, seq_len, d_head = keys.shape

        # Protect sink and recent tokens
        num_sink = self.config.num_sink
        num_recent = self.config.num_recent

        if seq_len <= num_sink + num_recent:
            if is_4d:
                keys = keys.view(batch_size, n_heads, seq_len, d_head).transpose(1, 2)
                keys = keys.transpose(1, 2)
                values = values.view(batch_size, n_heads, seq_len, d_head).transpose(1, 2)
                values = values.transpose(1, 2)
            return keys, values

        # Extract sink, middle, recent
        k_sink = keys[:, :num_sink, :]
        v_sink = values[:, :num_sink, :]
        k_recent = keys[:, -num_recent:, :] if num_recent > 0 else keys[:, seq_len:, :]
        v_recent = values[:, -num_recent:, :] if num_recent > 0 else values[:, seq_len:, :]

        middle_end = seq_len - num_recent if num_recent > 0 else seq_len
        k_middle = keys[:, num_sink:middle_end, :]
        v_middle = values[:, num_sink:middle_end, :]

        middle_len = k_middle.shape[1]
        if middle_len < self.merger_config.min_cluster_size:
            if is_4d:
                keys = keys.view(batch_size, n_heads, seq_len, d_head).transpose(1, 2)
                keys = keys.transpose(1, 2)
                values = values.view(batch_size, n_heads, seq_len, d_head).transpose(1, 2)
                values = values.transpose(1, 2)
            return keys, values

        # Compute adjacent similarities
        similarity = self._compute_adjacent_similarity(k_middle)

        # Identify merging sets
        merging_sets = self._identify_merging_sets(
            similarity, self.merger_config.similarity_threshold
        )

        # Process each batch element
        k_compressed_list = []
        v_compressed_list = []

        for b in range(k_middle.shape[0]):
            sets = merging_sets[b]

            if len(sets) == 0:
                # No merging needed
                k_compressed_list.append(k_middle[b:b+1])
                v_compressed_list.append(v_middle[b:b+1])
                continue

            # Build output by copying non-merged and inserting merged tokens
            k_out = []
            v_out = []
            prev_end = 0

            for start, end in sets:
                # Copy tokens before this set
                if start > prev_end:
                    k_out.append(k_middle[b, prev_end:start, :])
                    v_out.append(v_middle[b, prev_end:start, :])

                # Merge this set
                k_merged, v_merged = self._gaussian_kernel_merge(
                    k_middle[b:b+1], v_middle[b:b+1], start, end
                )
                k_out.append(k_merged)
                v_out.append(v_merged)

                prev_end = end

            # Copy remaining tokens
            if prev_end < middle_len:
                k_out.append(k_middle[b, prev_end:, :])
                v_out.append(v_middle[b, prev_end:, :])

            k_compressed = torch.cat(k_out, dim=0).unsqueeze(0)
            v_compressed = torch.cat(v_out, dim=0).unsqueeze(0)

            k_compressed_list.append(k_compressed)
            v_compressed_list.append(v_compressed)

        # Pad to same length and stack
        max_len = max(k.shape[1] for k in k_compressed_list)
        k_padded = []
        v_padded = []

        for k, v in zip(k_compressed_list, v_compressed_list):
            if k.shape[1] < max_len:
                pad_len = max_len - k.shape[1]
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
            k_padded.append(k)
            v_padded.append(v)

        k_middle_out = torch.cat(k_padded, dim=0)
        v_middle_out = torch.cat(v_padded, dim=0)

        # Concatenate sink + compressed_middle + recent
        keys_out = torch.cat([k_sink, k_middle_out, k_recent], dim=1)
        values_out = torch.cat([v_sink, v_middle_out, v_recent], dim=1)

        if is_4d:
            out_len = keys_out.shape[1]
            keys_out = keys_out.view(batch_size, n_heads, out_len, d_head)
            values_out = values_out.view(batch_size, n_heads, out_len, d_head)

        return keys_out, values_out


def kvmerger_compress(
    keys: Tensor,
    values: Tensor,
    similarity_threshold: float = 0.75,
    num_sink: int = 4,
    num_recent: int = 8,
) -> Tuple[Tensor, Tensor]:
    """Standalone KVMerger function for testing."""
    config = KVMergerConfig(
        num_sink=num_sink,
        num_recent=num_recent,
        similarity_threshold=similarity_threshold,
    )
    method = KVMerger(config)
    return method.compress(keys, values)
