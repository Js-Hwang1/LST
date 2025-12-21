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

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


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
        compressed_cache: Tuple[Tuple[Tensor, Tensor], ...],
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

        outputs = model(
            suffix_ids,
            past_key_values=compressed_cache,
            labels=labels,
        )

        return outputs.loss

    def forward_with_logits(
        self,
        model: nn.Module,
        suffix_ids: Tensor,
        compressed_cache: Tuple[Tuple[Tensor, Tensor], ...],
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute loss and return logits for analysis.

        Returns:
            Tuple of (loss, logits)
        """
        labels = suffix_ids.clone()

        if self.ignore_first_token:
            labels[:, 0] = -100

        outputs = model(
            suffix_ids,
            past_key_values=compressed_cache,
            labels=labels,
        )

        return outputs.loss, outputs.logits


def ppl_loss(
    model: nn.Module,
    suffix_ids: Tensor,
    compressed_cache: Tuple[Tuple[Tensor, Tensor], ...],
) -> Tensor:
    """Functional interface to PPLLoss."""
    loss_fn = PPLLoss()
    return loss_fn(model, suffix_ids, compressed_cache)
