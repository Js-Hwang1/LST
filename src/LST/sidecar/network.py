"""
LST Sidecar Network
===================

The complete sidecar network that maps windows of KV pairs to super-tokens.
This is the core compression module for LST.

Key architectural insights:
1. Hard norm projection is CRITICAL - without it, PPL explodes
2. Residual connection from mean baseline stabilizes training
3. Zero initialization of output projection for stable start
4. No dropout during training (for PPL optimization)
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import LSTConfig


class SidecarPPL(nn.Module):
    """
    Sidecar network for direct PPL training.

    Architecture:
        Input: (batch, window_size, 2*d_head) - concatenated K and V
        -> Linear projection to hidden_dim
        -> Positional embedding (learnable)
        -> TransformerEncoder (num_layers, 4 heads)
        -> Attention pooling with learned query
        -> Output projection to 2*d_head
        -> Add residual from window mean (scaled)
        -> Hard norm projection to match input norms

    The hard norm projection is CRITICAL:
    - Ensures compressed tokens have similar scale to original tokens
    - Prevents PPL explosion during inference
    - Acts as architectural constraint, not learned regularization
    """

    def __init__(
        self,
        d_head: int,
        window_size: int,
        hidden_dim: int = 256,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,  # CRITICAL: Use 0.0 for PPL training
    ):
        super().__init__()
        self.d_head = d_head
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(2 * d_head, hidden_dim)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, window_size, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,  # CRITICAL: No dropout for PPL training
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Attention pooling with learned query
        self.agg_query = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.5
        )
        self.agg_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Output projection (initialized to zero for stable start)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * d_head),
        )
        # Zero init final layer for stable training start
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Learnable residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        kv_window: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compress a window of KV pairs to a single super-token.

        Args:
            kv_window: (batch, window_size, 2*d_head) - concatenated K and V

        Returns:
            k_out: (batch, d_head) - compressed key
            v_out: (batch, d_head) - compressed value
        """
        B, W, _ = kv_window.shape

        # Compute mean for residual connection (baseline)
        kv_mean = kv_window.mean(dim=1)
        k_mean = kv_mean[:, : self.d_head]
        v_mean = kv_mean[:, self.d_head :]

        # Compute norms for hard projection (from input window)
        k_input = kv_window[:, :, : self.d_head]
        v_input = kv_window[:, :, self.d_head :]
        k_norm = k_input.norm(dim=-1).mean(dim=1, keepdim=True)  # (B, 1)
        v_norm = v_input.norm(dim=-1).mean(dim=1, keepdim=True)  # (B, 1)

        # Encode with transformer
        x = self.input_proj(kv_window) + self.pos_embed[:, :W, :]
        x = self.encoder(x)

        # Aggregate with attention pooling
        query = self.agg_query.expand(B, -1, -1)
        agg, _ = self.agg_attn(query, x, x)
        agg = agg.squeeze(1)  # (B, hidden_dim)

        # Project to K/V space
        residual = self.output_proj(agg)
        dk = residual[:, : self.d_head]
        dv = residual[:, self.d_head :]

        # Add residual from mean (learned scale)
        k_out = k_mean + self.residual_scale * dk
        v_out = v_mean + self.residual_scale * dv

        # CRITICAL: Hard norm projection
        # Ensures output has same norm as input tokens (average)
        k_out = k_out / (k_out.norm(dim=-1, keepdim=True) + 1e-8) * k_norm
        v_out = v_out / (v_out.norm(dim=-1, keepdim=True) + 1e-8) * v_norm

        return k_out, v_out


class Sidecar(nn.Module):
    """
    Generic sidecar wrapper with configurable architecture.

    This is a more flexible version that can use different encoders
    and aggregators. For direct PPL training, use SidecarPPL instead.
    """

    def __init__(self, config: LSTConfig):
        super().__init__()
        self.config = config
        self.d_head = config.d_head

        # Use SidecarPPL internally
        self.network = SidecarPPL(
            d_head=config.d_head,
            window_size=config.window_size,
            hidden_dim=config.encoder_hidden_dim,
            num_encoder_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            dropout=config.encoder_dropout,
        )

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compress a window of K, V pairs.

        Args:
            keys: (batch, window_size, d_head)
            values: (batch, window_size, d_head)

        Returns:
            k_out: (batch, d_head) - compressed key
            v_out: (batch, d_head) - compressed value
        """
        # Concatenate K and V
        kv_window = torch.cat([keys, values], dim=-1)
        return self.network(kv_window)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def compress_cache(
    cache: List[Tuple[Tensor, Tensor]],
    sidecar: nn.Module,
    window_size: int,
    num_sink: int = 4,
    num_recent: int = 8,
) -> List[Tuple[Tensor, Tensor]]:
    """
    Compress a full KV cache using the sidecar.

    Args:
        cache: List of (K, V) tuples per layer, each (1, n_heads, seq_len, d_head)
        sidecar: Trained sidecar module
        window_size: Window size for compression
        num_sink: Number of sink tokens to preserve
        num_recent: Number of recent tokens to preserve

    Returns:
        List of (K_compressed, V_compressed) tuples
    """
    num_layers = len(cache)
    device = cache[0][0].device
    dtype = cache[0][0].dtype

    new_cache = []

    for layer_idx in range(num_layers):
        k = cache[layer_idx][0]  # (1, H, S, D)
        v = cache[layer_idx][1]

        B, H, S, D = k.shape

        # Skip if not enough tokens to compress
        if S <= num_sink + num_recent + window_size:
            new_cache.append((k, v))
            continue

        # Split: sink, middle, recent
        ks = k[:, :, :num_sink, :]
        vs = v[:, :, :num_sink, :]
        kr = k[:, :, -num_recent:, :]
        vr = v[:, :, -num_recent:, :]
        km = k[:, :, num_sink:-num_recent, :]
        vm = v[:, :, num_sink:-num_recent, :]

        # Compute number of complete windows
        M = km.shape[2]
        num_windows = M // window_size
        trim_len = num_windows * window_size

        if num_windows == 0:
            new_cache.append((k, v))
            continue

        # Trim to complete windows
        km_trim = km[:, :, :trim_len, :]
        vm_trim = vm[:, :, :trim_len, :]
        km_leftover = km[:, :, trim_len:, :]
        vm_leftover = vm[:, :, trim_len:, :]

        # Reshape for sidecar: (B*H*num_windows, window_size, D)
        km_windows = km_trim.view(B, H, num_windows, window_size, D)
        vm_windows = vm_trim.view(B, H, num_windows, window_size, D)

        # Flatten batch and heads
        km_flat = km_windows.permute(0, 1, 2, 3, 4).reshape(-1, window_size, D)
        vm_flat = vm_windows.permute(0, 1, 2, 3, 4).reshape(-1, window_size, D)

        # Compress with sidecar
        kv_concat = torch.cat([km_flat, vm_flat], dim=-1)
        with torch.no_grad():
            k_comp, v_comp = sidecar.network(kv_concat)

        # Reshape back: (B, H, num_windows, D)
        k_comp = k_comp.view(B, H, num_windows, D)
        v_comp = v_comp.view(B, H, num_windows, D)

        # Concatenate: sink + compressed + leftover + recent
        k_new = torch.cat([ks, k_comp, km_leftover, kr], dim=2)
        v_new = torch.cat([vs, v_comp, vm_leftover, vr], dim=2)

        new_cache.append((k_new, v_new))

    return new_cache
