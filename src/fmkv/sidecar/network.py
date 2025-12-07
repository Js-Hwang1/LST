"""
Sidecar Network
===============

The complete Sidecar network Φ_θ that maps a window of N KV pairs
to a single coarse-grained super-token (K_CG, V_CG).

Architecture:
    Input:  (batch, N, 2*d_head) - concatenated K,V for window
    Encoder: Captures intra-window dependencies
    Aggregator: Compresses N -> 1
    Output Projection: Maps to K_CG, V_CG
    Output: (batch, 2*d_head) or (batch, d_head, 2)
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from fmkv.sidecar.config import SidecarConfig
from fmkv.sidecar.encoder import create_encoder
from fmkv.sidecar.aggregator import create_aggregator


class Sidecar(nn.Module):
    """
    The Sidecar network Φ_θ: Learned coarse-graining operator.
    
    Maps a window of fine-grained KV pairs to a single coarse-grained
    super-token that preserves gradient dynamics (force matching).
    
    Φ_θ: R^{N × 2d} → R^{1 × 2d}
    
    Args:
        config: SidecarConfig specifying architecture details.
    
    Example:
        >>> config = SidecarConfig(d_head=128, window_size=64)
        >>> sidecar = Sidecar(config)
        >>> 
        >>> # Input: batch of windows, each with N=64 tokens
        >>> # Each token has concatenated K and V vectors
        >>> kv_window = torch.randn(32, 64, 256)  # (batch, N, 2*d_head)
        >>> 
        >>> # Output: compressed super-tokens
        >>> k_cg, v_cg = sidecar(kv_window)
        >>> print(k_cg.shape)  # (32, 128)
        >>> print(v_cg.shape)  # (32, 128)
    """
    
    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.config = config
        
        # Create encoder
        self.encoder = create_encoder(config)
        
        # Create aggregator
        self.aggregator = create_aggregator(config)
        
        # Output projection to K_CG, V_CG
        if config.output_projection:
            # Create output projection with intermediate layer
            proj_hidden = nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim)
            proj_output = nn.Linear(config.encoder_hidden_dim, config.output_dim)
            
            # Tag final layer for special initialization (Bug #14 fix)
            proj_output._is_final_output = True
            
            self.output_proj = nn.Sequential(
                proj_hidden,
                nn.GELU(),
                proj_output,
            )
        else:
            # Direct projection if encoder_hidden_dim == output_dim
            if config.encoder_hidden_dim != config.output_dim:
                self.output_proj = nn.Linear(config.encoder_hidden_dim, config.output_dim)
                self.output_proj._is_final_output = True
            else:
                self.output_proj = nn.Identity()
        
        # Initialize weights (must be AFTER tagging layers)
        self.apply(self._init_weights)
        
        # Log parameter count
        self._num_parameters = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module: nn.Module):
        """
        Initialize weights using Xavier/Glorot initialization with careful scaling.
        
        Bug #14 Fix: Properly detect and initialize output projection layers.
        Bug #17 Fix: Initialize aggregator parameters (seeds, inducing_points, query).
        These were stuck at tiny values (* 0.02) causing zero outputs.
        """
        if isinstance(module, nn.Linear):
            # Check if this is tagged as final output layer
            is_final_output = getattr(module, '_is_final_output', False)
            
            if is_final_output:
                # Extra large gain for final output to prevent vanishing
                gain = 3.0
            else:
                # Use larger gain for all layers (helps gradient flow)
                gain = 2.0
            
            nn.init.xavier_uniform_(module.weight, gain=gain)
            
            if module.bias is not None:
                # Small positive bias to break symmetry
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        
        # Bug #17 Fix: Initialize aggregator learnable parameters
        # Check for specific parameter names that need proper initialization
        if hasattr(module, 'seeds'):
            # PMA pooling seeds - use larger std to prevent collapse
            nn.init.normal_(module.seeds, mean=0.0, std=0.5)
        if hasattr(module, 'inducing_points'):
            # ISAB inducing points - use larger std to prevent collapse
            nn.init.normal_(module.inducing_points, mean=0.0, std=0.5)
        if hasattr(module, 'query'):
            # Attention pooling query - use larger std to prevent collapse
            nn.init.normal_(module.query, mean=0.0, std=0.5)
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return self._num_parameters
    
    @property
    def device(self) -> torch.device:
        """Device of model parameters."""
        return next(self.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of model parameters."""
        return next(self.parameters()).dtype
    
    def forward(
        self,
        kv_window: Float[Tensor, "batch window_size input_dim"],
        attention_mask: Optional[Float[Tensor, "batch window_size"]] = None,
        return_split: bool = True,
    ) -> Union[Tuple[Float[Tensor, "batch d_head"], Float[Tensor, "batch d_head"]], Float[Tensor, "batch output_dim"]]:
        """
        Forward pass: compress window of KV pairs to single super-token.

        Args:
            kv_window: Input tensor of shape (batch, window_size, 2*d_head).
                       Contains concatenated [K, V] vectors for each position.
            attention_mask: Optional mask of shape (batch, window_size).
                           1 for valid positions, 0 for padding.
            return_split: If True, return (K_CG, V_CG) as separate tensors.
                         If False, return concatenated [K_CG; V_CG].

        Returns:
            If return_split=True:
                Tuple of (K_CG, V_CG), each of shape (batch, d_head)
            If return_split=False:
                Tensor of shape (batch, 2*d_head)
        """
        # Validate input shape
        batch_size, seq_len, input_dim = kv_window.shape
        assert input_dim == self.config.input_dim, (
            f"Expected input dim {self.config.input_dim}, got {input_dim}"
        )
        
        # Encode: capture intra-window dependencies
        # (batch, window_size, 2*d_head) -> (batch, window_size, encoder_hidden_dim)
        encoded = self.encoder(kv_window, attention_mask)
        
        # Aggregate: compress N -> 1
        # (batch, window_size, encoder_hidden_dim) -> (batch, encoder_hidden_dim)
        aggregated = self.aggregator(encoded, attention_mask)
        
        # Project to output dimension
        # (batch, encoder_hidden_dim) -> (batch, 2*d_head)
        output = self.output_proj(aggregated)
        
        if return_split:
            # Split into K_CG and V_CG
            k_cg, v_cg = output.chunk(2, dim=-1)
            return k_cg, v_cg
        else:
            return output
    
    def compress_cache(
        self,
        keys: Float[Tensor, "batch window_size d_head"],
        values: Float[Tensor, "batch window_size d_head"],
        attention_mask: Optional[Float[Tensor, "batch window_size"]] = None,
    ) -> Tuple[Float[Tensor, "batch d_head"], Float[Tensor, "batch d_head"]]:
        """
        Convenience method to compress separate K and V tensors.

        v4 Protocol: Hard Manifold Projection
        =====================================
        Architecturally enforces ||K_cg|| = R_K and ||V_cg|| = R_V where:
        - R_K = mean(||k_i||) over the input window
        - R_V = mean(||v_i||) over the input window

        This prevents both collapse (v1) and explosion (v2, v3) by design.

        Args:
            keys: Key tensor of shape (batch, window_size, d_head)
            values: Value tensor of shape (batch, window_size, d_head)
            attention_mask: Optional mask of shape (batch, window_size)

        Returns:
            Tuple of (K_CG, V_CG), each of shape (batch, d_head)
        """
        # Concatenate K and V
        kv_window = torch.cat([keys, values], dim=-1)

        # Get raw outputs from network
        k_raw, v_raw = self.forward(kv_window, attention_mask, return_split=True)

        # v4: Hard Manifold Projection
        # Compute target norms from input window (average over tokens)
        R_K = keys.norm(dim=-1).mean(dim=-1, keepdim=True)  # (batch, 1)
        R_V = values.norm(dim=-1).mean(dim=-1, keepdim=True)  # (batch, 1)

        # Normalize raw outputs and rescale to match input manifold
        k_cg = R_K * (k_raw / (k_raw.norm(dim=-1, keepdim=True) + 1e-8))
        v_cg = R_V * (v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-8))

        return k_cg, v_cg
    
    @torch.no_grad()
    def compress_cache_batched(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a full KV cache by processing windows.
        
        This method handles the windowing logic for compressing
        an entire cache during inference.
        
        Args:
            keys: Full key cache of shape (batch, seq_len, d_head)
            values: Full value cache of shape (batch, seq_len, d_head)
            window_size: Window size (defaults to config.window_size)
            stride: Stride between windows (defaults to window_size for non-overlapping)
        
        Returns:
            Tuple of compressed (keys, values), each of shape 
            (batch, num_windows, d_head)
        """
        if window_size is None:
            window_size = self.config.window_size
        if stride is None:
            stride = window_size  # Non-overlapping by default
        
        batch_size, seq_len, d_head = keys.shape
        
        # Calculate number of complete windows
        num_windows = (seq_len - window_size) // stride + 1
        
        compressed_keys = []
        compressed_values = []
        
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            
            k_window = keys[:, start:end, :]
            v_window = values[:, start:end, :]
            
            k_cg, v_cg = self.compress_cache(k_window, v_window)
            
            compressed_keys.append(k_cg)
            compressed_values.append(v_cg)
        
        # Stack along sequence dimension
        compressed_keys = torch.stack(compressed_keys, dim=1)
        compressed_values = torch.stack(compressed_values, dim=1)
        
        return compressed_keys, compressed_values
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio (window_size : 1)."""
        return self.config.window_size
    
    def __repr__(self) -> str:
        return (
            f"Sidecar(\n"
            f"  encoder={self.config.encoder_type.value},\n"
            f"  aggregator={self.config.aggregator_type.value},\n"
            f"  window_size={self.config.window_size},\n"
            f"  d_head={self.config.d_head},\n"
            f"  hidden_dim={self.config.encoder_hidden_dim},\n"
            f"  num_params={self.num_parameters:,}\n"
            f")"
        )


class MultiHeadSidecar(nn.Module):
    """
    Multi-head Sidecar for models with multiple attention heads.
    
    Option 1: Shared Sidecar - single network for all heads (parameter efficient)
    Option 2: Per-head Sidecar - separate network per head (more expressive)
    
    This implementation uses a shared Sidecar with head-specific
    input/output projections for a balance of efficiency and expressivity.
    """
    
    def __init__(
        self,
        config: SidecarConfig,
        share_across_heads: bool = True,
    ):
        super().__init__()
        self.config = config
        self.share_across_heads = share_across_heads
        self.n_heads = config.n_heads
        
        if share_across_heads:
            # Single shared sidecar
            self.sidecar = Sidecar(config)
        else:
            # Per-head sidecars (more parameters, potentially better)
            self.sidecars = nn.ModuleList([
                Sidecar(config) for _ in range(config.n_heads)
            ])
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache for all heads.
        
        Args:
            keys: Shape (batch, n_heads, window_size, d_head)
            values: Shape (batch, n_heads, window_size, d_head)
            attention_mask: Optional mask of shape (batch, window_size)
        
        Returns:
            Tuple of (K_CG, V_CG), each of shape (batch, n_heads, d_head)
        """
        batch_size, n_heads, window_size, d_head = keys.shape
        
        if self.share_across_heads:
            # Merge batch and head dimensions
            keys_flat = keys.view(batch_size * n_heads, window_size, d_head)
            values_flat = values.view(batch_size * n_heads, window_size, d_head)
            
            if attention_mask is not None:
                mask_flat = attention_mask.unsqueeze(1).expand(-1, n_heads, -1)
                mask_flat = mask_flat.reshape(batch_size * n_heads, window_size)
            else:
                mask_flat = None
            
            k_cg, v_cg = self.sidecar.compress_cache(keys_flat, values_flat, mask_flat)
            
            # Reshape back
            k_cg = k_cg.view(batch_size, n_heads, d_head)
            v_cg = v_cg.view(batch_size, n_heads, d_head)
        else:
            # Process each head separately
            k_cg_list = []
            v_cg_list = []
            
            for h in range(n_heads):
                k_h, v_h = self.sidecars[h].compress_cache(
                    keys[:, h, :, :],
                    values[:, h, :, :],
                    attention_mask,
                )
                k_cg_list.append(k_h)
                v_cg_list.append(v_h)
            
            k_cg = torch.stack(k_cg_list, dim=1)
            v_cg = torch.stack(v_cg_list, dim=1)
        
        return k_cg, v_cg
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

