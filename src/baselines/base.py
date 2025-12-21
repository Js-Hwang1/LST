"""
Base class for KV Cache Compression Methods
============================================

All compression methods inherit from this base class to ensure
consistent interfaces for evaluation and comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass
class CompressionConfig:
    """Base configuration for compression methods."""

    # Compression parameters
    num_sink: int = 4  # Number of sink tokens to keep
    num_recent: int = 8  # Number of recent tokens to keep
    budget: int | None = None  # Total token budget (if specified)

    # Model parameters (for reference)
    d_head: int = 64  # Dimension per attention head
    n_heads: int = 32  # Number of attention heads

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.num_sink >= 0, "num_sink must be non-negative"
        assert self.num_recent >= 0, "num_recent must be non-negative"


class CompressionMethod(ABC):
    """
    Abstract base class for KV cache compression methods.

    All baseline implementations must inherit from this class and
    implement the `compress` method for fair comparison.

    Interface contract:
    - Input: Full KV cache (keys, values) for a single layer
    - Output: Compressed KV cache respecting the budget constraint
    - Methods must be deterministic for reproducibility
    """

    def __init__(self, config: CompressionConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the method."""
        pass

    @abstractmethod
    def compress(
        self,
        keys: Tensor,
        values: Tensor,
        attention_scores: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress a KV cache to the specified budget.

        Args:
            keys: Key tensor of shape (batch, seq_len, d_head) or
                  (batch, n_heads, seq_len, d_head)
            values: Value tensor with same shape as keys
            attention_scores: Optional attention scores for importance-based
                              methods. Shape varies by method.
            **kwargs: Method-specific additional arguments

        Returns:
            Tuple of (compressed_keys, compressed_values)
            Output shapes depend on the compression strategy
        """
        pass

    def get_budget(self, seq_len: int) -> int:
        """
        Calculate the token budget for a given sequence length.

        Args:
            seq_len: Original sequence length

        Returns:
            Number of tokens to retain after compression
        """
        if self.config.budget is not None:
            return min(self.config.budget, seq_len)
        return min(self.config.num_sink + self.config.num_recent, seq_len)

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"


def gather_indices(
    tensor: Tensor,
    indices: Tensor,
    dim: int = 1,
) -> Tensor:
    """
    Gather elements from tensor along specified dimension using indices.

    Utility function for index-based token selection.

    Args:
        tensor: Source tensor to gather from
        indices: Indices to gather
        dim: Dimension to gather along

    Returns:
        Gathered tensor
    """
    if tensor.dim() == 3:
        # (batch, seq, d_head)
        batch_size, _, d_head = tensor.shape
        budget = indices.shape[-1]
        idx_expanded = indices.unsqueeze(-1).expand(batch_size, budget, d_head)
        return tensor.gather(dim=dim, index=idx_expanded)

    elif tensor.dim() == 4:
        # (batch, n_heads, seq, d_head)
        batch_size, n_heads, _, d_head = tensor.shape
        budget = indices.shape[-1]
        idx_expanded = indices.unsqueeze(1).unsqueeze(-1)
        idx_expanded = idx_expanded.expand(batch_size, n_heads, budget, d_head)
        return tensor.gather(dim=dim + 1, index=idx_expanded)

    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
