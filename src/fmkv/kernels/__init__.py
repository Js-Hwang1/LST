"""
CUDA/Triton Kernels
===================

High-performance kernels for Force-Matched KV compression:

- attention_jacobian: Fused Jacobian computation ∂Attn/∂q
- softmax_jacobian: Specialized softmax Jacobian kernel
- force_matching: Fused force matching loss computation

These kernels provide 3-10x speedup over naive PyTorch implementations
on A100/H100 GPUs.

Usage:
    from fmkv.kernels import triton_attention_jacobian
    
    jacobian = triton_attention_jacobian(query, keys, values)
"""

import torch

# Check Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None

from fmkv.kernels.attention_jacobian import (
    compute_attention_jacobian_fused,
    AttentionJacobianKernel,
)

# Export appropriate implementation based on availability
if TRITON_AVAILABLE:
    from fmkv.kernels.triton_jacobian import (
        triton_attention_jacobian,
        triton_attention_jacobian_batched,
        triton_force_matching_loss,
    )
else:
    # Fallback to PyTorch implementation
    from fmkv.losses.jacobian import (
        compute_attention_jacobian as triton_attention_jacobian,
        compute_attention_jacobian_batched as triton_attention_jacobian_batched,
    )
    triton_force_matching_loss = None

__all__ = [
    "TRITON_AVAILABLE",
    "compute_attention_jacobian_fused",
    "AttentionJacobianKernel",
    "triton_attention_jacobian",
    "triton_attention_jacobian_batched",
    "triton_force_matching_loss",
]

