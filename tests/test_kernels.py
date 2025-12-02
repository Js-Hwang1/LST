"""
Tests for CUDA/Triton Kernels
=============================

Tests for high-performance Jacobian computation kernels.
Run with: pytest tests/test_kernels.py -v
"""

import pytest
import torch

from fmkv.kernels import (
    TRITON_AVAILABLE,
    compute_attention_jacobian_fused,
    AttentionJacobianKernel,
)
from fmkv.losses.jacobian import compute_attention_jacobian


class TestAttentionJacobianFused:
    """Tests for fused attention Jacobian computation."""
    
    def test_single_query(self):
        """Test with single query."""
        batch_size = 4
        seq_len = 16
        d_head = 32
        
        query = torch.randn(batch_size, d_head)
        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)
        
        jacobian = compute_attention_jacobian_fused(query, keys, values)
        
        assert jacobian.shape == (batch_size, d_head, d_head)
        assert torch.isfinite(jacobian).all()
    
    def test_batched_queries(self):
        """Test with multiple queries."""
        batch_size = 4
        num_queries = 8
        seq_len = 16
        d_head = 32
        
        queries = torch.randn(batch_size, num_queries, d_head)
        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)
        
        jacobians = compute_attention_jacobian_fused(queries, keys, values)
        
        assert jacobians.shape == (batch_size, num_queries, d_head, d_head)
        assert torch.isfinite(jacobians).all()
    
    def test_matches_reference(self):
        """Verify fused implementation matches reference."""
        batch_size = 2
        seq_len = 8
        d_head = 16
        
        query = torch.randn(batch_size, d_head)
        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)
        
        # Fused implementation
        jac_fused = compute_attention_jacobian_fused(query, keys, values)
        
        # Reference implementation
        jac_ref = compute_attention_jacobian(query, keys, values)
        
        # Should match closely
        assert torch.allclose(jac_fused, jac_ref, rtol=1e-3, atol=1e-3)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the computation."""
        batch_size = 2
        seq_len = 8
        d_head = 16
        
        query = torch.randn(batch_size, d_head, requires_grad=True)
        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)
        
        jacobian = compute_attention_jacobian_fused(query, keys, values)
        loss = jacobian.sum()
        
        # Should be able to backprop (even if gradients are None)
        loss.backward()


class TestAttentionJacobianKernel:
    """Tests for the kernel module wrapper."""
    
    def test_kernel_creation(self):
        """Test kernel can be created."""
        kernel = AttentionJacobianKernel(d_head=64)
        assert kernel.d_head == 64
        assert kernel.scale == 64 ** -0.5
    
    def test_kernel_forward(self):
        """Test kernel forward pass."""
        d_head = 32
        kernel = AttentionJacobianKernel(d_head=d_head, use_triton=False)
        
        batch_size = 4
        seq_len = 16
        
        query = torch.randn(batch_size, d_head)
        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)
        
        jacobian = kernel(query, keys, values)
        
        assert jacobian.shape == (batch_size, d_head, d_head)
    
    def test_kernel_chunked(self):
        """Test chunked processing for memory efficiency."""
        d_head = 32
        kernel = AttentionJacobianKernel(d_head=d_head, use_triton=False, chunk_size=4)
        
        batch_size = 2
        num_queries = 16
        seq_len = 8
        
        queries = torch.randn(batch_size, num_queries, d_head)
        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)
        
        jacobians = kernel(queries, keys, values)
        
        assert jacobians.shape == (batch_size, num_queries, d_head, d_head)


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestTritonKernels:
    """Tests for Triton kernels (requires Triton and CUDA)."""
    
    @pytest.fixture
    def cuda_tensors(self):
        """Create CUDA tensors for testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        batch_size = 4
        seq_len = 32
        d_head = 64
        
        return {
            "query": torch.randn(batch_size, d_head, device="cuda"),
            "keys": torch.randn(batch_size, seq_len, d_head, device="cuda"),
            "values": torch.randn(batch_size, seq_len, d_head, device="cuda"),
        }
    
    def test_triton_single_query(self, cuda_tensors):
        """Test Triton kernel with single query."""
        from fmkv.kernels import triton_attention_jacobian
        
        jacobian = triton_attention_jacobian(
            cuda_tensors["query"],
            cuda_tensors["keys"],
            cuda_tensors["values"],
        )
        
        assert jacobian.shape == (4, 64, 64)
        assert torch.isfinite(jacobian).all()
    
    def test_triton_batched(self, cuda_tensors):
        """Test Triton kernel with batched queries."""
        from fmkv.kernels import triton_attention_jacobian_batched
        
        queries = torch.randn(4, 8, 64, device="cuda")
        
        jacobians = triton_attention_jacobian_batched(
            queries,
            cuda_tensors["keys"],
            cuda_tensors["values"],
        )
        
        assert jacobians.shape == (4, 8, 64, 64)
        assert torch.isfinite(jacobians).all()
    
    def test_triton_matches_pytorch(self, cuda_tensors):
        """Verify Triton kernel matches PyTorch implementation."""
        from fmkv.kernels import triton_attention_jacobian
        
        # Triton result
        jac_triton = triton_attention_jacobian(
            cuda_tensors["query"],
            cuda_tensors["keys"],
            cuda_tensors["values"],
        )
        
        # PyTorch result
        jac_pytorch = compute_attention_jacobian_fused(
            cuda_tensors["query"],
            cuda_tensors["keys"],
            cuda_tensors["values"],
        )
        
        # Should match closely
        assert torch.allclose(jac_triton, jac_pytorch, rtol=1e-2, atol=1e-2)


class TestNumericalStability:
    """Tests for numerical stability of kernel computations."""
    
    def test_large_values(self):
        """Test with large input values."""
        batch_size = 2
        seq_len = 8
        d_head = 16
        
        query = torch.randn(batch_size, d_head) * 100
        keys = torch.randn(batch_size, seq_len, d_head) * 100
        values = torch.randn(batch_size, seq_len, d_head) * 100
        
        jacobian = compute_attention_jacobian_fused(query, keys, values)
        
        assert torch.isfinite(jacobian).all()
    
    def test_small_values(self):
        """Test with small input values."""
        batch_size = 2
        seq_len = 8
        d_head = 16
        
        query = torch.randn(batch_size, d_head) * 1e-4
        keys = torch.randn(batch_size, seq_len, d_head) * 1e-4
        values = torch.randn(batch_size, seq_len, d_head) * 1e-4
        
        jacobian = compute_attention_jacobian_fused(query, keys, values)
        
        assert torch.isfinite(jacobian).all()
    
    def test_long_sequence(self):
        """Test with long sequences."""
        batch_size = 2
        seq_len = 512
        d_head = 64
        
        query = torch.randn(batch_size, d_head)
        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)
        
        jacobian = compute_attention_jacobian_fused(query, keys, values)
        
        assert jacobian.shape == (batch_size, d_head, d_head)
        assert torch.isfinite(jacobian).all()


class TestDtypes:
    """Test different data types."""
    
    def test_float32(self):
        """Test with float32."""
        query = torch.randn(2, 32, dtype=torch.float32)
        keys = torch.randn(2, 8, 32, dtype=torch.float32)
        values = torch.randn(2, 8, 32, dtype=torch.float32)
        
        jacobian = compute_attention_jacobian_fused(query, keys, values)
        assert jacobian.dtype == torch.float32
    
    def test_float64(self):
        """Test with float64 for higher precision."""
        query = torch.randn(2, 32, dtype=torch.float64)
        keys = torch.randn(2, 8, 32, dtype=torch.float64)
        values = torch.randn(2, 8, 32, dtype=torch.float64)
        
        jacobian = compute_attention_jacobian_fused(query, keys, values)
        assert jacobian.dtype == torch.float64
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_float16_cuda(self):
        """Test with float16 on CUDA."""
        query = torch.randn(2, 32, dtype=torch.float16, device="cuda")
        keys = torch.randn(2, 8, 32, dtype=torch.float16, device="cuda")
        values = torch.randn(2, 8, 32, dtype=torch.float16, device="cuda")
        
        jacobian = compute_attention_jacobian_fused(query, keys, values)
        assert jacobian.dtype == torch.float16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

