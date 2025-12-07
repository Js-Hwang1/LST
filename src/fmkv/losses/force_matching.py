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
    """Configuration for Force Matching Loss.

    v4 Protocol: Hard Manifold Projection
    ======================================

    v1: force=0.0 (disabled) -> Jacobian collapse (ratio=0.02)
    v2: force=10.0, norm=0.1 -> Scale explosion (ratio=24.25)
    v3: direction=5.0, magnitude=1.0, manifold=2.0 -> WORSE explosion (ratio=105.5)
        Problem: anti_collapse penalty dominated loss (65%), driving explosion

    v4 Fix: ARCHITECTURAL enforcement of manifold constraint
    - Hard projection layer in Sidecar: ||K_cg|| = R_K, ||V_cg|| = R_V
    - Remove ALL regularization losses (manifold, anti_collapse, magnitude, etc.)
    - Keep ONLY: L_force_cos + lambda * L_consistency

    Loss function:
        L_v4 = L_force_cos + lambda * L_consistency
        where L_force_cos = 1 - cos(J_dense, J_cg)
    """

    # Number of future queries to sample for force matching
    num_future_queries: int = 16

    # === v4 Loss Weights (Hard Manifold Projection) ===
    #
    # Only TWO non-zero weights:
    # 1. force_direction_weight: Match Jacobian direction (steering)
    # 2. consistency_weight: Match attention output (validation)

    # lambda_1: Consistency (output MSE) - ensures compressed output matches dense
    consistency_weight: float = 1.0

    # lambda_2: Force DIRECTION matching (cosine similarity) - PRIMARY objective
    # L_dir = 1 - cos(J_dense, J_cg)
    force_direction_weight: float = 1.0

    # === ALL OTHER WEIGHTS SET TO 0.0 (v4 uses hard projection instead) ===

    # v3 weights - DISABLED (now handled by hard projection)
    force_magnitude_weight: float = 0.0  # Hard projection ensures ||K_cg|| = R_K
    manifold_weight: float = 0.0  # Hard projection ensures manifold constraint
    output_magnitude_weight: float = 0.0  # Not needed with hard projection
    kv_match_weight: float = 0.0  # Not needed - we match Jacobians directly
    diversity_weight: float = 0.0  # Let network learn diversity naturally

    # Legacy v2 weights - DISABLED
    force_matching_weight: float = 0.0
    kv_norm_weight: float = 0.0
    jacobian_norm_weight: float = 0.0  # No anti-collapse needed with hard projection
    min_jacobian_ratio: float = 0.1  # Not used in v4

    # Normalization
    normalize_jacobian: bool = True

    # Gradient clipping for stability
    jacobian_clip_value: Optional[float] = 50.0


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
        forces: Optional[torch.Tensor] = None,  # Pre-collected KV position gradients (batch, seq, hidden)
        force_queries: Optional[torch.Tensor] = None,  # Pre-collected query gradients (batch, num_q, hidden)
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
            forces: Optional pre-collected KV position gradients, shape (batch, window_size, hidden_dim).
                    If provided, uses force magnitudes as importance weights for KV matching.
                    Tokens with higher gradient norms are weighted more heavily.
            force_queries: Optional pre-collected query gradients, shape (batch, num_queries, hidden_dim).
                    These are the TRUE force matching targets: F = ∇_q L with dense attention.
                    If provided, uses force magnitudes as importance weights for consistency loss.
                    Queries with higher gradient norms (more sensitive) are matched more carefully.

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
        
        # === v3 Force Matching: Cosine-Magnitude Decomposition ===
        #
        # v2 Problem: Raw MSE || J_dense - J_cg ||^2 caused scale explosion
        #             because the sidecar amplified ||K||, ||V|| to match aggregate Jacobian.
        #
        # v3 Solution: Decouple DIRECTION (steering) from MAGNITUDE (intensity)
        #
        # L_direction = 1 - cos(J_dense, J_cg)  -- Match the steering direction
        # L_magnitude = |log(||J_dense||) - log(||J_cg||)|  -- Match intensity in log-space

        # Flatten Jacobians for cosine similarity: (batch, d*d)
        dense_jac_flat = dense_jacobian.view(batch_size, -1)  # (batch, d*d)
        cg_jac_flat = cg_jacobian.view(batch_size, -1)  # (batch, d*d)

        # Compute norms
        dense_jac_norm = dense_jac_flat.norm(dim=-1)  # (batch,)
        cg_jac_norm = cg_jac_flat.norm(dim=-1)  # (batch,)

        # === L_direction: Cosine Similarity Loss ===
        # cos(J_dense, J_cg) = (J_dense · J_cg) / (||J_dense|| * ||J_cg||)
        dot_product = (dense_jac_flat * cg_jac_flat).sum(dim=-1)  # (batch,)
        cosine_sim = dot_product / (dense_jac_norm * cg_jac_norm + 1e-8)
        force_direction_loss = 1.0 - cosine_sim  # 0 when perfectly aligned, 2 when opposite

        # === L_magnitude: Log-Space Magnitude Matching ===
        # |log(||J_dense||) - log(||J_cg||)| prevents both collapse and explosion
        # log-space makes gradient symmetric around ratio=1.0
        log_dense_norm = torch.log(dense_jac_norm + 1e-8)
        log_cg_norm = torch.log(cg_jac_norm + 1e-8)
        force_magnitude_loss = torch.abs(log_dense_norm - log_cg_norm)

        # Legacy v2 MSE loss (kept for metrics, weight=0 by default)
        jacobian_diff = dense_jacobian - cg_jacobian
        dense_jac_sq = (dense_jacobian ** 2).sum(dim=(-2, -1)).detach() + 1e-8
        force_matching_loss = (jacobian_diff ** 2).sum(dim=(-2, -1)) / dense_jac_sq
        
        # === Consistency Loss ===
        # || Attn_dense - Attn_cg ||^2 (preserve output magnitude)
        # TRUE Force Matching Enhancement: Weight by query gradient magnitudes
        # Queries with higher ∇_q L (more sensitive to attention changes) get higher weights
        output_diff = dense_outputs - cg_outputs  # (batch, num_q, d_head)
        per_query_mse = (output_diff ** 2).sum(dim=-1)  # (batch, num_q)

        if force_queries is not None:
            # force_queries: (batch, num_q, hidden_dim) - may have different dim than d_head
            # Use gradient magnitude as importance weight per query
            query_importance = force_queries.norm(dim=-1)  # (batch, num_q)
            query_importance = query_importance / (query_importance.sum(dim=-1, keepdim=True) + 1e-8)
            # Weighted MSE: sensitive queries matter more
            consistency_loss = (per_query_mse * query_importance).sum(dim=-1)  # (batch,)
        else:
            # Standard unweighted average
            consistency_loss = per_query_mse.sum(dim=-1) / queries.size(1)  # (batch,)

        consistency_loss = consistency_loss / self.d_head  # Normalize by dimension
        
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

        # Bug #31 Fix: Use RELATIVE norm matching (consistent with force matching)
        # Old: (||J_dense|| - ||J_cg||)² / d²  (arbitrary normalization)
        # New: ((||J_dense|| - ||J_cg||) / ||J_dense||)²  (relative error)
        jac_ratio = cg_jac_norm / (dense_jac_norm + 1e-8)
        jacobian_norm_loss = ((1.0 - jac_ratio) ** 2)  # 0 when matched, 1 when collapsed

        # Penalty when ratio drops below threshold (hard floor)
        collapse_penalty = F.relu(self.config.min_jacobian_ratio - jac_ratio) ** 2

        # Combined anti-collapse loss
        anti_collapse_loss = jacobian_norm_loss + collapse_penalty

        # === Bug #33 Fix: Direct KV Matching Loss ===
        # Anchor super-tokens to window means - this is CRITICAL for good compression!
        # Mean pooling gets 5% error, so this provides a strong baseline.
        # The force matching loss can then refine beyond mean pooling.
        #
        # Bug #37 Enhancement: Force-Weighted KV Matching
        # When forces are provided, use force magnitudes as importance weights.
        # Tokens with higher gradient norms (more "force" on generation) get weighted more.
        # This is the TRUE force matching approach: preserve high-gradient tokens better.
        if forces is not None:
            # forces: (batch, seq, hidden_dim) - compute magnitude per token
            force_weights = forces.norm(dim=-1)  # (batch, seq)
            force_weights = force_weights + 1e-8  # Prevent division by zero
            force_weights = force_weights / force_weights.sum(dim=-1, keepdim=True)  # Normalize

            # Force-weighted mean: tokens with higher gradient norms weighted more
            k_window_mean = (keys * force_weights.unsqueeze(-1)).sum(dim=1)  # (batch, d_head)
            v_window_mean = (values * force_weights.unsqueeze(-1)).sum(dim=1)  # (batch, d_head)
        else:
            # Standard mean pooling (fallback)
            k_window_mean = keys.mean(dim=1)  # (batch, d_head)
            v_window_mean = values.mean(dim=1)  # (batch, d_head)

        # Handle multi-window case: k_cg may be (batch, num_windows, d) or (batch, d)
        if k_cg.dim() == 3:
            # Multi-window: compute mean of all super-tokens and match to overall window mean
            k_cg_mean = k_cg.mean(dim=1)  # (batch, d)
            v_cg_mean = v_cg.mean(dim=1)
        else:
            k_cg_mean = k_cg  # (batch, d)
            v_cg_mean = v_cg

        # Bug #34 Fix: Use RELATIVE MSE for K and V separately
        # Problem: V has ~60x smaller magnitude than K, so raw MSE ignores V
        # Solution: Normalize each component's MSE by its target norm
        k_norm_sq = (k_window_mean ** 2).sum(dim=-1) + 1e-8  # (batch,)
        v_norm_sq = (v_window_mean ** 2).sum(dim=-1) + 1e-8  # (batch,)

        k_match_loss = ((k_cg_mean - k_window_mean) ** 2).sum(dim=-1) / k_norm_sq  # Relative MSE
        v_match_loss = ((v_cg_mean - v_window_mean) ** 2).sum(dim=-1) / v_norm_sq  # Relative MSE

        # Average K and V losses (now equally weighted)
        kv_match_loss = (k_match_loss + v_match_loss) / 2

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

        # === v3 Strict Manifold Regularization ===
        # Forces compressed tokens to reside on the same Riemannian manifold as dense tokens.
        #
        # v2 used: L_norm = (||k_cg|| - μ_K)² + (||v_cg|| - μ_V)²  (difference-based)
        #          This allowed explosion: ||k_cg|| = 30.6 vs μ_K = 19.1
        #
        # v3 uses: L_manifold = (||k_cg||/μ_K - 1)² + (||v_cg||/μ_V - 1)²  (ratio-based)
        #          This is symmetric and prevents BOTH collapse AND explosion.
        #
        # The ratio-based formulation has constant gradient magnitude around ratio=1.0

        k_dense_norm_mean = keys.norm(dim=-1).mean(dim=-1)  # (batch,) - μ_K
        v_dense_norm_mean = values.norm(dim=-1).mean(dim=-1)  # (batch,) - μ_V

        # Handle multi-window case
        if k_cg.dim() == 3:
            k_cg_norm_mean = k_cg.norm(dim=-1).mean(dim=-1)  # (batch,)
            v_cg_norm_mean = v_cg.norm(dim=-1).mean(dim=-1)  # (batch,)
        else:
            k_cg_norm_mean = k_cg.norm(dim=-1)  # (batch,)
            v_cg_norm_mean = v_cg.norm(dim=-1)  # (batch,)

        # v3: Ratio-based manifold loss
        # L_manifold = (||k_cg||/μ_K - 1)² + (||v_cg||/μ_V - 1)²
        k_ratio = k_cg_norm_mean / (k_dense_norm_mean + 1e-8)
        v_ratio = v_cg_norm_mean / (v_dense_norm_mean + 1e-8)
        manifold_loss = (k_ratio - 1.0).pow(2) + (v_ratio - 1.0).pow(2)

        # Legacy v2 loss (for comparison metrics)
        kv_norm_loss = (k_cg_norm_mean - k_dense_norm_mean).pow(2) + (v_cg_norm_mean - v_dense_norm_mean).pow(2)

        # === Total Loss ===
        # v3: Cosine-Magnitude decomposition with strict manifold constraint
        total_loss = (
            # v3 Primary: Direction + Magnitude decomposition
            self.config.force_direction_weight * force_direction_loss +
            self.config.force_magnitude_weight * force_magnitude_loss +
            self.config.manifold_weight * manifold_loss +
            # Base requirements
            self.config.consistency_weight * consistency_loss +
            self.config.output_magnitude_weight * magnitude_loss +
            self.config.kv_match_weight * kv_match_loss +
            self.config.diversity_weight * diversity_loss +
            # Legacy v2 (weights = 0 by default)
            self.config.force_matching_weight * force_matching_loss +
            self.config.jacobian_norm_weight * anti_collapse_loss +
            self.config.kv_norm_weight * kv_norm_loss
        )
        
        # Reduction
        if reduction == "mean":
            total_loss = total_loss.mean()
            # v3 losses
            force_direction_loss = force_direction_loss.mean()
            force_magnitude_loss = force_magnitude_loss.mean()
            manifold_loss = manifold_loss.mean()
            # Base losses
            force_matching_loss = force_matching_loss.mean()
            consistency_loss = consistency_loss.mean()
            magnitude_loss = magnitude_loss.mean()
            anti_collapse_loss = anti_collapse_loss.mean()
            jacobian_norm_loss = jacobian_norm_loss.mean()
            collapse_penalty = collapse_penalty.mean()
            diversity_loss = diversity_loss.mean()
            kv_match_loss = kv_match_loss.mean()
            kv_norm_loss = kv_norm_loss.mean()
            # Ratios
            jac_ratio = jac_ratio.mean()
            k_ratio = k_ratio.mean()
            v_ratio = v_ratio.mean()
            cosine_sim = cosine_sim.mean()
        elif reduction == "sum":
            total_loss = total_loss.sum()
            force_direction_loss = force_direction_loss.sum()
            force_magnitude_loss = force_magnitude_loss.sum()
            manifold_loss = manifold_loss.sum()
            force_matching_loss = force_matching_loss.sum()
            consistency_loss = consistency_loss.sum()
            magnitude_loss = magnitude_loss.sum()
            anti_collapse_loss = anti_collapse_loss.sum()
            jacobian_norm_loss = jacobian_norm_loss.sum()
            collapse_penalty = collapse_penalty.sum()
            diversity_loss = diversity_loss.sum()
            kv_match_loss = kv_match_loss.sum()
            kv_norm_loss = kv_norm_loss.sum()
            # Ratios still mean
            jac_ratio = jac_ratio.mean()
            k_ratio = k_ratio.mean()
            v_ratio = v_ratio.mean()
            cosine_sim = cosine_sim.mean()

        metrics = {
            "loss/total": total_loss.detach(),
            # v3 Primary losses
            "loss/force_direction": force_direction_loss.detach(),
            "loss/force_magnitude": force_magnitude_loss.detach(),
            "loss/manifold": manifold_loss.detach(),
            # Base losses
            "loss/force_matching": force_matching_loss.detach(),
            "loss/consistency": consistency_loss.detach(),
            "loss/magnitude": magnitude_loss.detach(),
            "loss/anti_collapse": anti_collapse_loss.detach(),
            "loss/jacobian_norm": jacobian_norm_loss.detach(),
            "loss/collapse_penalty": collapse_penalty.detach(),
            "loss/diversity": diversity_loss.detach(),
            "loss/kv_match": kv_match_loss.detach(),
            "loss/kv_norm": kv_norm_loss.detach(),
            # v3 Key metrics
            "jacobian/cosine_sim": cosine_sim.detach(),  # Should approach 1.0
            "jacobian/dense_norm": dense_jac_norm.mean().detach(),
            "jacobian/cg_norm": cg_jac_norm.mean().detach(),
            "jacobian/ratio": jac_ratio.detach(),  # v3 target: 0.5 - 1.5
            # Manifold metrics
            "manifold/k_ratio": k_ratio.detach(),  # v3 target: ~1.0
            "manifold/v_ratio": v_ratio.detach(),  # v3 target: ~1.0
            # Output norms
            "output/dense_norm": dense_norm.mean().detach(),
            "output/cg_norm": cg_norm.mean().detach(),
            # KV norms (absolute values)
            "kv_norm/k_dense": k_dense_norm_mean.mean().detach(),
            "kv_norm/k_cg": k_cg_norm_mean.mean().detach(),
            "kv_norm/v_dense": v_dense_norm_mean.mean().detach(),
            "kv_norm/v_cg": v_cg_norm_mean.mean().detach(),
            "force/using_force_weights": torch.tensor(force_queries is not None, dtype=torch.float32),
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

