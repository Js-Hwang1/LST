"""
Diversity Loss for Super-Token Training
========================================

Prevents mode collapse where all super-tokens become identical.

Problem:
    Without diversity pressure, the sidecar may learn to produce the same
    super-token for all windows (e.g., the global mean). This loses information.

Solution:
    Contrastive learning: super-tokens from different windows should be
    distinguishable, while representing their source window faithfully.

Implementation:
    InfoNCE-style loss where each super-token should be most similar to its
    source window's representation (positive) and dissimilar to other windows
    (negatives).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiversityLoss(nn.Module):
    """
    Contrastive diversity loss for super-tokens.

    Ensures different windows produce distinguishable super-tokens.

    Args:
        temperature: Softmax temperature for contrastive loss
        mode: 'infonce' or 'cosine'
    """

    def __init__(
        self,
        temperature: float = 0.1,
        mode: str = "infonce",
    ):
        super().__init__()
        self.temperature = temperature
        self.mode = mode

    def forward(
        self,
        super_tokens: Tensor,
        window_representations: Tensor = None,
    ) -> Tensor:
        """
        Compute diversity loss.

        Args:
            super_tokens: Compressed representations (N, D) where N is number
                          of windows across batch
            window_representations: Optional original window representations (N, D)
                                   If None, uses super_tokens for self-contrast

        Returns:
            Diversity loss scalar
        """
        N, D = super_tokens.shape

        if N <= 1:
            return torch.tensor(0.0, device=super_tokens.device)

        # Normalize for cosine similarity
        super_tokens_norm = F.normalize(super_tokens, dim=-1)

        if self.mode == "infonce":
            return self._infonce_loss(super_tokens_norm, window_representations)
        elif self.mode == "cosine":
            return self._cosine_diversity_loss(super_tokens_norm)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _infonce_loss(
        self,
        super_tokens: Tensor,
        window_representations: Tensor = None,
    ) -> Tensor:
        """
        InfoNCE contrastive loss.

        Each super-token should be most similar to its source window.
        """
        N = super_tokens.shape[0]

        if window_representations is None:
            # Self-contrast: use augmented view or just ensure diversity
            # Here we just ensure super-tokens are different from each other
            sim_matrix = super_tokens @ super_tokens.T  # (N, N)

            # We want diagonal to be high, off-diagonal to be low
            # But without positive pairs, we just minimize off-diagonal
            mask = torch.eye(N, device=super_tokens.device, dtype=torch.bool)
            off_diag = sim_matrix[~mask].view(N, N - 1)

            # Minimize mean of off-diagonal similarities
            loss = off_diag.mean()

        else:
            # With window representations: standard InfoNCE
            window_norm = F.normalize(window_representations, dim=-1)
            sim_matrix = super_tokens @ window_norm.T / self.temperature

            # Diagonal should be high (super-token matches its window)
            labels = torch.arange(N, device=super_tokens.device)
            loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def _cosine_diversity_loss(self, super_tokens: Tensor) -> Tensor:
        """
        Simple cosine diversity: minimize pairwise similarity.

        This encourages super-tokens to be as different as possible.
        """
        N = super_tokens.shape[0]

        # Pairwise cosine similarity
        sim_matrix = super_tokens @ super_tokens.T  # (N, N)

        # Mask out diagonal
        mask = torch.eye(N, device=super_tokens.device, dtype=torch.bool)
        off_diag = sim_matrix[~mask]

        # Penalize high similarity between different super-tokens
        # Using squared similarity to penalize high values more
        loss = (off_diag ** 2).mean()

        return loss


class WindowRepresentationLoss(nn.Module):
    """
    Auxiliary loss ensuring super-tokens represent their source windows.

    The super-token should allow "reconstruction" of the window's information.
    We don't literally reconstruct, but ensure the super-token is informative
    about the window content.

    Implementation: Super-token should be similar to window mean (baseline)
    plus have low reconstruction error when used as a summary.
    """

    def __init__(self, reconstruction_weight: float = 0.1):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight

    def forward(
        self,
        super_tokens: Tensor,
        window_means: Tensor,
    ) -> Tensor:
        """
        Compute window representation loss.

        Args:
            super_tokens: (N, D) compressed tokens
            window_means: (N, D) mean of each source window

        Returns:
            Loss scalar
        """
        # Super-token should be informative about window
        # Cosine similarity between super-token and window mean
        super_norm = F.normalize(super_tokens, dim=-1)
        mean_norm = F.normalize(window_means, dim=-1)

        similarity = (super_norm * mean_norm).sum(dim=-1)

        # We want high similarity, so minimize (1 - similarity)
        loss = (1 - similarity).mean()

        return loss


def diversity_loss(
    super_tokens: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Functional interface to DiversityLoss."""
    loss_fn = DiversityLoss(temperature=temperature, mode="cosine")
    return loss_fn(super_tokens)
