"""
Query-Probing Attention Alignment Loss (QPAA)
=============================================

Novel training objective for KV cache compression.

Key Insight:
    PPL training optimizes for specific continuations, but the compressed
    cache will be queried by arbitrary future tokens. We need super-tokens
    that preserve attention behavior for ANY query.

Method:
    1. Sample random "probe queries" during training
    2. Compute attention output using dense cache: out_dense = Attn(q, K_dense, V_dense)
    3. Compute attention output using compressed cache: out_comp = Attn(q, K_comp, V_comp)
    4. Minimize ||out_dense - out_comp||^2

Why This Works:
    - Random queries approximate the distribution of all possible future queries
    - If attention works for random queries, it generalizes to real queries
    - Similar principle to dropout regularization or data augmentation

Why This is Novel:
    - Prior work on KV compression doesn't use query probing
    - Jacobian matching (prior work) uses ∂y/∂q, but is unstable
    - Our approach directly matches attention outputs, which is what matters

Theoretical Justification:
    Let A(q) = softmax(qK^T/√d) V be the attention output.
    We want: A_compressed(q) ≈ A_dense(q) for all q

    By sampling q ~ N(0, I) and minimizing E_q[||A_dense(q) - A_comp(q)||^2],
    we ensure the compressed cache preserves the attention operator's behavior
    across the entire query space.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class QueryProbingLoss(nn.Module):
    """
    Query-Probing Attention Alignment loss.

    Ensures compressed KV cache produces same attention output as dense cache
    for randomly sampled probe queries.

    Args:
        num_probes: Number of random queries to sample per batch
        probe_distribution: Distribution for probe queries ('normal', 'uniform', 'sphere')
        temperature: Softmax temperature for attention
        detach_dense: Whether to detach dense outputs (recommended: True)
    """

    def __init__(
        self,
        num_probes: int = 8,
        probe_distribution: str = "sphere",
        temperature: float = 1.0,
        detach_dense: bool = True,
    ):
        super().__init__()
        self.num_probes = num_probes
        self.probe_distribution = probe_distribution
        self.temperature = temperature
        self.detach_dense = detach_dense

    def sample_probes(
        self,
        batch_size: int,
        num_heads: int,
        d_head: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Sample random probe queries.

        Args:
            batch_size: Batch size
            num_heads: Number of attention heads
            d_head: Dimension per head
            device: Device
            dtype: Data type

        Returns:
            Probe queries of shape (num_probes, batch_size, num_heads, 1, d_head)
        """
        shape = (self.num_probes, batch_size, num_heads, 1, d_head)

        if self.probe_distribution == "normal":
            probes = torch.randn(shape, device=device, dtype=dtype)
        elif self.probe_distribution == "uniform":
            probes = torch.rand(shape, device=device, dtype=dtype) * 2 - 1
        elif self.probe_distribution == "sphere":
            # Unit sphere: better coverage of query space
            probes = torch.randn(shape, device=device, dtype=dtype)
            probes = probes / (probes.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            raise ValueError(f"Unknown probe distribution: {self.probe_distribution}")

        return probes

    def attention_output(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        """
        Compute scaled dot-product attention output.

        Args:
            query: (B, H, 1, D) or (B, H, Q, D)
            keys: (B, H, S, D)
            values: (B, H, S, D)

        Returns:
            Attention output of shape (B, H, Q, D)
        """
        d_head = query.shape[-1]
        scale = math.sqrt(d_head) * self.temperature

        # Compute attention scores
        attn_weights = torch.matmul(query, keys.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute output
        output = torch.matmul(attn_weights, values)

        return output

    def forward(
        self,
        k_dense: Tensor,
        v_dense: Tensor,
        k_compressed: Tensor,
        v_compressed: Tensor,
        reduction: str = "mean",
    ) -> Tensor:
        """
        Compute Query-Probing Attention Alignment loss.

        Args:
            k_dense: Dense keys (B, H, S_dense, D)
            v_dense: Dense values (B, H, S_dense, D)
            k_compressed: Compressed keys (B, H, S_comp, D)
            v_compressed: Compressed values (B, H, S_comp, D)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Loss scalar (or tensor if reduction='none')
        """
        B, H, S_dense, D = k_dense.shape
        device = k_dense.device
        dtype = k_dense.dtype

        # Sample probe queries
        probes = self.sample_probes(B, H, D, device, dtype)

        losses = []

        for probe_idx in range(self.num_probes):
            q = probes[probe_idx]  # (B, H, 1, D)

            # Attention output with dense cache
            out_dense = self.attention_output(q, k_dense, v_dense)

            # Attention output with compressed cache
            out_comp = self.attention_output(q, k_compressed, v_compressed)

            # Detach dense to avoid backprop through frozen model
            if self.detach_dense:
                out_dense = out_dense.detach()

            # MSE loss between outputs
            loss = F.mse_loss(out_comp, out_dense, reduction="none")
            losses.append(loss)

        # Stack and reduce
        all_losses = torch.stack(losses, dim=0)  # (num_probes, B, H, 1, D)

        if reduction == "mean":
            return all_losses.mean()
        elif reduction == "sum":
            return all_losses.sum()
        else:
            return all_losses


def query_probing_loss(
    k_dense: Tensor,
    v_dense: Tensor,
    k_compressed: Tensor,
    v_compressed: Tensor,
    num_probes: int = 8,
) -> Tensor:
    """Functional interface to QueryProbingLoss."""
    loss_fn = QueryProbingLoss(num_probes=num_probes)
    return loss_fn(k_dense, v_dense, k_compressed, v_compressed)
