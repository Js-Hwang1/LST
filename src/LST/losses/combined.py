"""
Combined Training Loss for LST
==============================

Multi-objective loss combining:
    1. PPL Loss: Ensures generation quality
    2. Query-Probing Loss (QPAA): Ensures attention works for arbitrary queries
    3. Diversity Loss: Prevents super-token collapse

The combined loss is:
    L = λ_ppl * L_ppl + λ_qpaa * L_qpaa + λ_div * L_diversity

Default weights are tuned for stable training:
    - λ_ppl = 1.0 (primary objective)
    - λ_qpaa = 0.5 (secondary, regularization)
    - λ_div = 0.1 (tertiary, prevents collapse)

Training Strategy:
    1. Start with PPL-only for first 10% of training (warm-up)
    2. Gradually introduce QPAA and diversity losses
    3. This allows the sidecar to first learn basic compression,
       then refine for query-invariance
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from .ppl import PPLLoss
from .query_probing import QueryProbingLoss
from .diversity import DiversityLoss


@dataclass
class LossWeights:
    """Weights for combined loss components."""

    ppl: float = 1.0
    query_probing: float = 0.5
    diversity: float = 0.1

    def scale_auxiliary(self, factor: float) -> "LossWeights":
        """Scale auxiliary losses (for warmup)."""
        return LossWeights(
            ppl=self.ppl,
            query_probing=self.query_probing * factor,
            diversity=self.diversity * factor,
        )


class CombinedLoss(nn.Module):
    """
    Combined training loss for LST.

    Combines PPL, Query-Probing, and Diversity losses.

    Args:
        weights: Loss weights
        num_probes: Number of random probes for QPAA
        diversity_temperature: Temperature for contrastive diversity
        warmup_steps: Steps before auxiliary losses are fully enabled
    """

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        num_probes: int = 8,
        diversity_temperature: float = 0.1,
        warmup_steps: int = 0,
    ):
        super().__init__()

        self.weights = weights or LossWeights()
        self.warmup_steps = warmup_steps

        self.ppl_loss = PPLLoss()
        self.query_probing_loss = QueryProbingLoss(
            num_probes=num_probes,
            probe_distribution="sphere",
        )
        self.diversity_loss = DiversityLoss(
            temperature=diversity_temperature,
            mode="cosine",
        )

    def get_weights(self, step: int) -> LossWeights:
        """Get loss weights for current step (with warmup)."""
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return self.weights

        # Linear warmup for auxiliary losses
        factor = step / self.warmup_steps
        return self.weights.scale_auxiliary(factor)

    def forward(
        self,
        model: nn.Module,
        suffix_ids: Tensor,
        compressed_cache: Tuple[Tuple[Tensor, Tensor], ...],
        dense_cache: Tuple[Tuple[Tensor, Tensor], ...],
        super_tokens: Optional[Tensor] = None,
        step: int = 0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            model: Language model
            suffix_ids: Token IDs for continuation (B, S)
            compressed_cache: Compressed KV cache
            dense_cache: Original dense KV cache (for QPAA)
            super_tokens: Optional super-token embeddings for diversity loss
            step: Current training step (for warmup)

        Returns:
            Tuple of (total_loss, loss_dict with individual components)
        """
        weights = self.get_weights(step)
        loss_dict = {}

        # 1. PPL Loss
        l_ppl = self.ppl_loss(model, suffix_ids, compressed_cache)
        loss_dict["ppl"] = l_ppl.item()

        total_loss = weights.ppl * l_ppl

        # 2. Query-Probing Loss (QPAA)
        if weights.query_probing > 0 and len(dense_cache) > 0:
            l_qpaa = self._compute_qpaa_loss(dense_cache, compressed_cache)
            loss_dict["qpaa"] = l_qpaa.item()
            total_loss = total_loss + weights.query_probing * l_qpaa

        # 3. Diversity Loss
        if weights.diversity > 0 and super_tokens is not None:
            l_div = self.diversity_loss(super_tokens)
            loss_dict["diversity"] = l_div.item()
            total_loss = total_loss + weights.diversity * l_div

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict

    def _compute_qpaa_loss(
        self,
        dense_cache: Tuple[Tuple[Tensor, Tensor], ...],
        compressed_cache: Tuple[Tuple[Tensor, Tensor], ...],
    ) -> Tensor:
        """Compute QPAA loss averaged across layers."""
        losses = []

        for (k_dense, v_dense), (k_comp, v_comp) in zip(dense_cache, compressed_cache):
            # Skip if shapes don't make sense
            if k_dense.shape[2] == k_comp.shape[2]:
                continue  # No compression happened

            loss = self.query_probing_loss(k_dense, v_dense, k_comp, v_comp)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=dense_cache[0][0].device)

        return torch.stack(losses).mean()


def combined_loss(
    model: nn.Module,
    suffix_ids: Tensor,
    compressed_cache: Tuple[Tuple[Tensor, Tensor], ...],
    dense_cache: Tuple[Tuple[Tensor, Tensor], ...],
    super_tokens: Optional[Tensor] = None,
    weights: Optional[LossWeights] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Functional interface to CombinedLoss."""
    loss_fn = CombinedLoss(weights=weights)
    return loss_fn(model, suffix_ids, compressed_cache, dense_cache, super_tokens)
