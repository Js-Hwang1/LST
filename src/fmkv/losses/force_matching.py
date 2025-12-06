"""
Force Matching Loss
===================

The core loss function for training the Sidecar network.

Inspired by Force Matching in Molecular Dynamics, this loss ensures
that the coarse-grained super-token exerts the same "gradient force"
on future queries as the original fine-grained token cluster.

L_FM(θ) = Σ_q || Σ_j ∂Attn(q, k_j, v_j)/∂q - ∂Attn(q, K_CG, V_CG)/∂q ||_F^2

Where:
    - q: Future query vectors (sampled from training data)
    - k_j, v_j: Original fine-grained KV pairs in the window
    - K_CG, V_CG: Coarse-grained super-token produced by Sidecar
    - ||·||_F: Frobenius norm
"""

from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from fmkv.losses.jacobian import (
    compute_attention_jacobian,
    compute_attention_jacobian_batched,
    compute_aggregate_jacobian,
    compute_attention_output,
)


@dataclass
class ForceMatchingLossConfig:
    """Configuration for Force Matching Loss."""

    # Number of future queries to sample for force matching
    num_future_queries: int = 16

    # Loss weighting (Bug #8 Fix: Increased magnitude weight)
    force_matching_weight: float = 1.0
    consistency_weight: float = 0.1
    output_magnitude_weight: float = 0.2  # Increased from 0.05 to prevent output collapse

    # Bug #29 Fix: Anti-collapse regularization
    # Penalizes when cg_jacobian norm falls below dense_jacobian norm
    jacobian_norm_weight: float = 0.5  # Weight for Jacobian norm matching
    min_jacobian_ratio: float = 0.1  # Minimum allowed ratio of cg_norm / dense_norm

    # Bug #30 Fix: Diversity loss to prevent window collapse
    # When multi-window training produces similar outputs, Jacobian -> 0
    diversity_weight: float = 1.0  # Weight for diversity regularization

    # Normalization
    normalize_jacobian: bool = True

    # Gradient clipping for stability (Bug #6 Fix: Increased from 10.0)
    jacobian_clip_value: Optional[float] = 50.0  # Increased to allow larger gradients


class ForceMatchingLoss(nn.Module):
    """
    Force Matching Loss for training the Sidecar network.
    
    This loss matches the attention Jacobians (gradient forces) between:
    1. Dense attention over the full token window
    2. Compressed attention using the Sidecar's super-token
    
    The intuition: if the Jacobians match, the compressed cache will
    "steer" the model's generation in the same direction as the original.
    
    Args:
        config: ForceMatchingLossConfig with loss parameters
        d_head: Dimension of attention heads
    
    Example:
        >>> loss_fn = ForceMatchingLoss(d_head=128)
        >>> 
        >>> # Original window of KV pairs
        >>> keys = torch.randn(32, 64, 128)      # (batch, window, d_head)
        >>> values = torch.randn(32, 64, 128)
        >>> 
        >>> # Compressed by Sidecar
        >>> k_cg = torch.randn(32, 128)          # (batch, d_head)
        >>> v_cg = torch.randn(32, 128)
        >>> 
        >>> # Future queries that would attend to this window
        >>> queries = torch.randn(32, 16, 128)   # (batch, num_q, d_head)
        >>> 
        >>> loss, metrics = loss_fn(queries, keys, values, k_cg, v_cg)
    """
    
    def __init__(
        self,
        d_head: int,
        config: Optional[ForceMatchingLossConfig] = None,
    ):
        super().__init__()
        self.d_head = d_head
        self.config = config or ForceMatchingLossConfig()
        self.scale = d_head ** -0.5
    
    def compute_dense_jacobian(
        self,
        queries: Float[Tensor, "batch num_queries d_head"],
        keys: Float[Tensor, "batch seq_len d_head"],
        values: Float[Tensor, "batch seq_len d_head"],
    ) -> Tuple[Float[Tensor, "batch d_head d_head"], Float[Tensor, "batch num_queries d_head"]]:
        """
        Compute the aggregate Jacobian for dense (uncompressed) attention.

        For dense attention, we compute the Jacobian w.r.t. each key-value
        pair and sum them (representing total force from the window).

        Args:
            queries: Shape (batch, num_queries, d_head)
            keys: Shape (batch, seq_len, d_head)
            values: Shape (batch, seq_len, d_head)

        Returns:
            Tuple of:
                - aggregate_jacobian: Shape (batch, d_head, d_head)
                - attention_outputs: Shape (batch, num_queries, d_head)
        """
        batch_size, num_queries, d = queries.shape
        seq_len = keys.size(1)
        
        # Compute Jacobians for all queries
        # This gives us ∂Attn(q, K, V)/∂q for each query
        jacobians = compute_attention_jacobian_batched(
            queries, keys, values, self.scale
        )  # (batch, num_q, d, d)
        
        # Sum over queries to get aggregate force
        aggregate_jacobian = jacobians.sum(dim=1)  # (batch, d, d)
        
        # Also compute attention outputs for consistency loss
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attention_outputs = torch.matmul(attn_weights, values)  # (batch, num_q, d)
        
        return aggregate_jacobian, attention_outputs
    
    def compute_compressed_jacobian(
        self,
        queries: Float[Tensor, "batch num_queries d_head"],
        k_cg: Float[Tensor, "batch *num_compressed d_head"],
        v_cg: Float[Tensor, "batch *num_compressed d_head"],
    ) -> Tuple[Float[Tensor, "batch d_head d_head"], Float[Tensor, "batch num_queries d_head"]]:
        """
        Compute the Jacobian for compressed attention.

        Bug #19 Fix: k_cg and v_cg can now have multiple super-tokens (seq_len > 1).
        This makes attention non-trivial and Jacobian non-zero.

        Args:
            queries: Shape (batch, num_queries, d_head)
            k_cg: Compressed keys, shape (batch, num_compressed, d_head) or (batch, d_head)
            v_cg: Compressed values, shape (batch, num_compressed, d_head) or (batch, d_head)

        Returns:
            Tuple of:
                - aggregate_jacobian: Shape (batch, d_head, d_head)
                - attention_outputs: Shape (batch, num_queries, d_head)
        """
        # Handle both single super-token and multiple super-tokens
        if k_cg.dim() == 2:
            k_cg = k_cg.unsqueeze(1)  # (batch, 1, d)
            v_cg = v_cg.unsqueeze(1)  # (batch, 1, d)
        # Otherwise k_cg/v_cg already have shape (batch, num_compressed, d)
        
        # Compute Jacobians - now non-trivial if num_compressed > 1
        jacobians = compute_attention_jacobian_batched(
            queries, k_cg, v_cg, self.scale
        )  # (batch, num_q, d, d)
        
        # Sum over queries
        aggregate_jacobian = jacobians.sum(dim=1)  # (batch, d, d)
        
        # Attention output - now uses real softmax over multiple tokens
        scores = torch.matmul(queries, k_cg.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attention_outputs = torch.matmul(attn_weights, v_cg)  # (batch, num_q, d)
        
        return aggregate_jacobian, attention_outputs
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        k_cg: torch.Tensor,
        v_cg: torch.Tensor,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the Force Matching loss.
        
        Args:
            queries: Future query vectors, shape (batch, num_queries, d_head)
            keys: Original key matrix, shape (batch, window_size, d_head)
            values: Original value matrix, shape (batch, window_size, d_head)
            k_cg: Compressed key from Sidecar, shape (batch, d_head)
            v_cg: Compressed value from Sidecar, shape (batch, d_head)
            reduction: "mean", "sum", or "none"
        
        Returns:
            Tuple of:
                - total_loss: Scalar loss tensor
                - metrics: Dict with individual loss components
        """
        batch_size = queries.size(0)
        
        # Compute dense (original) Jacobian and outputs
        dense_jacobian, dense_outputs = self.compute_dense_jacobian(
            queries, keys, values
        )
        
        # Compute compressed Jacobian and outputs
        cg_jacobian, cg_outputs = self.compute_compressed_jacobian(
            queries, k_cg, v_cg
        )
        
        # Clip Jacobians for stability
        if self.config.jacobian_clip_value is not None:
            dense_jacobian = torch.clamp(
                dense_jacobian,
                -self.config.jacobian_clip_value,
                self.config.jacobian_clip_value,
            )
            cg_jacobian = torch.clamp(
                cg_jacobian,
                -self.config.jacobian_clip_value,
                self.config.jacobian_clip_value,
            )
        
        # === Force Matching Loss ===
        # || J_dense - J_cg ||_F^2
        jacobian_diff = dense_jacobian - cg_jacobian
        force_matching_loss = (jacobian_diff ** 2).sum(dim=(-2, -1))  # (batch,)
        
        if self.config.normalize_jacobian:
            # Normalize by matrix dimension
            force_matching_loss = force_matching_loss / (self.d_head ** 2)
        
        # === Consistency Loss ===
        # || Attn_dense - Attn_cg ||^2 (preserve output magnitude)
        output_diff = dense_outputs - cg_outputs
        consistency_loss = (output_diff ** 2).sum(dim=(-2, -1))  # (batch,)
        consistency_loss = consistency_loss / (queries.size(1) * self.d_head)
        
        # === Output Magnitude Loss ===
        # Encourage compressed output to have similar norm to dense output
        dense_norm = dense_outputs.norm(dim=-1).mean(dim=-1)  # (batch,)
        cg_norm = cg_outputs.norm(dim=-1).mean(dim=-1)  # (batch,)
        magnitude_loss = (dense_norm - cg_norm) ** 2

        # === Bug #29 Fix: Jacobian Norm Anti-Collapse Loss ===
        # Prevent the cg_jacobian from collapsing to zero
        # Compute Jacobian norms
        dense_jac_norm = dense_jacobian.norm(dim=(-2, -1))  # (batch,)
        cg_jac_norm = cg_jacobian.norm(dim=(-2, -1))  # (batch,)

        # Two-part anti-collapse:
        # 1. Match Jacobian norms (encourage cg to have similar magnitude as dense)
        jacobian_norm_loss = (dense_jac_norm - cg_jac_norm) ** 2 / (self.d_head ** 2)

        # 2. Penalty when cg_norm ratio drops below threshold
        # This provides a hard floor to prevent complete collapse
        jac_ratio = cg_jac_norm / (dense_jac_norm + 1e-8)
        collapse_penalty = F.relu(self.config.min_jacobian_ratio - jac_ratio) ** 2
        # Scale penalty to be significant
        collapse_penalty = collapse_penalty * dense_jac_norm.detach() ** 2 / (self.d_head ** 2)

        # Combined anti-collapse loss
        anti_collapse_loss = jacobian_norm_loss + collapse_penalty

        # === Bug #30 Fix: Diversity Loss ===
        # Prevent compressed tokens from collapsing to identical outputs
        # When k_cg tokens are identical, softmax becomes uniform, Jacobian -> 0
        diversity_loss = torch.zeros(batch_size, device=k_cg.device, dtype=k_cg.dtype)

        if k_cg.dim() == 3 and k_cg.size(1) > 1:
            # k_cg: (batch, num_windows, d_head)
            num_windows = k_cg.size(1)

            # Normalize for cosine similarity
            k_cg_norm = F.normalize(k_cg, dim=-1)  # (batch, num_windows, d_head)
            v_cg_norm = F.normalize(v_cg, dim=-1)

            # Compute pairwise cosine similarity matrix
            # (batch, num_windows, num_windows)
            k_sim = torch.bmm(k_cg_norm, k_cg_norm.transpose(-2, -1))
            v_sim = torch.bmm(v_cg_norm, v_cg_norm.transpose(-2, -1))

            # We want OFF-DIAGONAL elements to be small (diverse outputs)
            # Create mask to exclude diagonal
            eye = torch.eye(num_windows, device=k_cg.device, dtype=k_cg.dtype)
            off_diag_mask = 1.0 - eye  # (num_windows, num_windows)

            # Penalize high similarity between different windows
            # Use squared similarity to make gradient stronger near 1
            k_off_diag = (k_sim * off_diag_mask).pow(2).sum(dim=(-2, -1))
            v_off_diag = (v_sim * off_diag_mask).pow(2).sum(dim=(-2, -1))

            # Normalize by number of off-diagonal pairs
            num_pairs = num_windows * (num_windows - 1)
            diversity_loss = (k_off_diag + v_off_diag) / (2 * num_pairs)

        # === Total Loss ===
        total_loss = (
            self.config.force_matching_weight * force_matching_loss +
            self.config.consistency_weight * consistency_loss +
            self.config.output_magnitude_weight * magnitude_loss +
            self.config.jacobian_norm_weight * anti_collapse_loss +
            self.config.diversity_weight * diversity_loss
        )
        
        # Reduction
        if reduction == "mean":
            total_loss = total_loss.mean()
            force_matching_loss = force_matching_loss.mean()
            consistency_loss = consistency_loss.mean()
            magnitude_loss = magnitude_loss.mean()
            anti_collapse_loss = anti_collapse_loss.mean()
            jacobian_norm_loss = jacobian_norm_loss.mean()
            collapse_penalty = collapse_penalty.mean()
            diversity_loss = diversity_loss.mean()
            jac_ratio = jac_ratio.mean()
        elif reduction == "sum":
            total_loss = total_loss.sum()
            force_matching_loss = force_matching_loss.sum()
            consistency_loss = consistency_loss.sum()
            magnitude_loss = magnitude_loss.sum()
            anti_collapse_loss = anti_collapse_loss.sum()
            jacobian_norm_loss = jacobian_norm_loss.sum()
            collapse_penalty = collapse_penalty.sum()
            diversity_loss = diversity_loss.sum()
            jac_ratio = jac_ratio.mean()  # Still mean for ratio

        metrics = {
            "loss/total": total_loss.detach(),
            "loss/force_matching": force_matching_loss.detach(),
            "loss/consistency": consistency_loss.detach(),
            "loss/magnitude": magnitude_loss.detach(),
            "loss/anti_collapse": anti_collapse_loss.detach(),
            "loss/jacobian_norm": jacobian_norm_loss.detach(),
            "loss/collapse_penalty": collapse_penalty.detach(),
            "loss/diversity": diversity_loss.detach(),  # Bug #30: Track window diversity
            "jacobian/dense_norm": dense_jac_norm.mean().detach(),
            "jacobian/cg_norm": cg_jac_norm.mean().detach(),
            "jacobian/ratio": jac_ratio.detach(),  # Key metric to watch!
            "output/dense_norm": dense_norm.mean().detach(),
            "output/cg_norm": cg_norm.mean().detach(),
        }
        
        return total_loss, metrics


class PerQueryForceMatchingLoss(nn.Module):
    """
    Per-query Force Matching Loss variant.
    
    Instead of aggregating Jacobians, this matches individual query
    Jacobians. May provide more fine-grained training signal.
    """
    
    def __init__(self, d_head: int, normalize: bool = True):
        super().__init__()
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.normalize = normalize
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        k_cg: torch.Tensor,
        v_cg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute per-query Force Matching loss.
        
        L = (1/|Q|) Σ_q || ∂Attn(q, K, V)/∂q - ∂Attn(q, K_CG, V_CG)/∂q ||_F^2
        """
        # Dense Jacobians: (batch, num_q, d, d)
        dense_jacobians = compute_attention_jacobian_batched(
            queries, keys, values, self.scale
        )
        
        # Compressed Jacobians
        k_cg = k_cg.unsqueeze(1)
        v_cg = v_cg.unsqueeze(1)
        cg_jacobians = compute_attention_jacobian_batched(
            queries, k_cg, v_cg, self.scale
        )
        
        # Per-query Frobenius distance
        diff = dense_jacobians - cg_jacobians
        per_query_loss = (diff ** 2).sum(dim=(-2, -1))  # (batch, num_q)
        
        if self.normalize:
            per_query_loss = per_query_loss / (self.d_head ** 2)
        
        # Average over queries and batch
        loss = per_query_loss.mean()
        
        metrics = {
            "loss/per_query_fm": loss.detach(),
            "loss/per_query_std": per_query_loss.std().detach(),
        }
        
        return loss, metrics


class HierarchicalForceMatchingLoss(nn.Module):
    """
    Hierarchical Force Matching Loss for multi-scale compression.
    
    Matches forces at multiple scales:
    1. Individual token level
    2. Sub-window level
    3. Full window level
    
    This can help the Sidecar learn both local and global dynamics.
    """
    
    def __init__(
        self,
        d_head: int,
        scales: Tuple[int, ...] = (4, 16, 64),
        scale_weights: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        self.d_head = d_head
        self.scales = scales
        self.scale_weights = scale_weights or tuple(1.0 / len(scales) for _ in scales)
        self.scale = d_head ** -0.5
        
        assert len(self.scale_weights) == len(scales)
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        k_cg: torch.Tensor,
        v_cg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hierarchical force matching loss at multiple scales.
        """
        batch_size, window_size, d = keys.shape
        total_loss = torch.tensor(0.0, device=keys.device, dtype=keys.dtype)
        metrics = {}
        
        for scale, weight in zip(self.scales, self.scale_weights):
            if scale > window_size:
                continue
            
            # For this scale, average over sub-windows
            num_subwindows = window_size // scale
            
            if num_subwindows == 1:
                # Full window - just use aggregate Jacobian
                dense_jac = compute_aggregate_jacobian(queries, keys, values, self.scale)
                k_exp = k_cg.unsqueeze(1)
                v_exp = v_cg.unsqueeze(1)
                cg_jac = compute_aggregate_jacobian(queries, k_exp, v_exp, self.scale)
            else:
                # Average Jacobians over sub-windows
                subwindow_jacs = []
                for i in range(num_subwindows):
                    start = i * scale
                    end = start + scale
                    sub_keys = keys[:, start:end, :]
                    sub_values = values[:, start:end, :]
                    jac = compute_aggregate_jacobian(queries, sub_keys, sub_values, self.scale)
                    subwindow_jacs.append(jac)
                
                dense_jac = torch.stack(subwindow_jacs, dim=1).mean(dim=1)
                
                # CG should match the average behavior
                k_exp = k_cg.unsqueeze(1)
                v_exp = v_cg.unsqueeze(1)
                cg_jac = compute_aggregate_jacobian(queries, k_exp, v_exp, self.scale)
            
            # Loss at this scale
            diff = dense_jac - cg_jac
            scale_loss = (diff ** 2).sum(dim=(-2, -1)).mean()
            scale_loss = scale_loss / (self.d_head ** 2)
            
            total_loss = total_loss + weight * scale_loss
            metrics[f"loss/scale_{scale}"] = scale_loss.detach()
        
        metrics["loss/hierarchical_total"] = total_loss.detach()
        
        return total_loss, metrics

