"""
Perplexity Loss for LST Training
=================================

Standard perplexity loss computed on continuation after cache compression.
This ensures the compressed cache maintains generation quality.

The loss is computed as:
    L_ppl = CrossEntropy(model(suffix | compressed_prefix), suffix)

This is the "downstream task" loss - it measures whether the compressed
cache actually works for language modeling.
"""

import torch.nn as nn
from torch import Tensor

try:
    from transformers.cache_utils import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False


def convert_to_cache(past_key_values):
    """Convert tuple of (K, V) to DynamicCache for newer transformers."""
    if not HAS_DYNAMIC_CACHE:
        return past_key_values
    if past_key_values is None:
        return None
    # If already a Cache object, return as-is
    if hasattr(past_key_values, 'get_seq_length'):
        return past_key_values
    # Convert from legacy tuple format
    return DynamicCache.from_legacy_cache(past_key_values)


class PPLLoss(nn.Module):
    """
    Perplexity loss for cache compression.

    Computes cross-entropy loss on suffix given compressed prefix cache.

    Args:
        ignore_first_token: Whether to ignore first suffix token in loss
    """

    def __init__(self, ignore_first_token: bool = True):
        super().__init__()
        self.ignore_first_token = ignore_first_token

    def forward(
        self,
        model: nn.Module,
        suffix_ids: Tensor,
        compressed_cache: tuple[tuple[Tensor, Tensor], ...],
    ) -> Tensor:
        """
        Compute perplexity loss.

        Args:
            model: Language model with forward(input_ids, past_key_values, labels)
            suffix_ids: Token IDs for suffix (B, S_suffix)
            compressed_cache: Compressed KV cache from prefix

        Returns:
            Cross-entropy loss (scalar)
        """
        labels = suffix_ids.clone()

        if self.ignore_first_token:
            labels[:, 0] = -100  # Ignore first token

        # Convert to Cache object for newer transformers versions
        cache = convert_to_cache(compressed_cache)

        outputs = model(
            suffix_ids,
            past_key_values=cache,
            labels=labels,
        )

        return outputs.loss

    def forward_with_logits(
        self,
        model: nn.Module,
        suffix_ids: Tensor,
        compressed_cache: tuple[tuple[Tensor, Tensor], ...],
    ) -> tuple[Tensor, Tensor]:
        """
        Compute loss and return logits for analysis.

        Returns:
            Tuple of (loss, logits)
        """
        labels = suffix_ids.clone()

        if self.ignore_first_token:
            labels[:, 0] = -100

        # Convert to Cache object for newer transformers versions
        cache = convert_to_cache(compressed_cache)

        outputs = model(
            suffix_ids,
            past_key_values=cache,
            labels=labels,
        )

        return outputs.loss, outputs.logits


def ppl_loss(
    model: nn.Module,
    suffix_ids: Tensor,
    compressed_cache: tuple[tuple[Tensor, Tensor], ...],
) -> Tensor:
    """Functional interface to PPLLoss."""
    loss_fn = PPLLoss()
    return loss_fn(model, suffix_ids, compressed_cache)
