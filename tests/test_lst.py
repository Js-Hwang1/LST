"""
Tests for LST (Learned Super-Token) module.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.LST import LSTConfig, Sidecar, SidecarPPL


class TestLSTConfig:
    """Tests for LSTConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LSTConfig()
        assert config.d_head == 128
        assert config.window_size == 8
        assert config.encoder_dropout == 0.0  # Critical for PPL training

    def test_input_output_dim(self):
        """Test input/output dimension properties."""
        config = LSTConfig(d_head=64)
        assert config.input_dim == 128  # 2 * d_head
        assert config.output_dim == 128


class TestSidecarPPL:
    """Tests for SidecarPPL network."""

    @pytest.fixture
    def sidecar(self):
        """Create a small sidecar for testing."""
        return SidecarPPL(
            d_head=64,
            window_size=8,
            hidden_dim=128,
            num_encoder_layers=1,
        )

    def test_forward_shape(self, sidecar):
        """Test forward pass output shape."""
        batch_size = 4
        window_size = 8
        d_head = 64

        kv_window = torch.randn(batch_size, window_size, 2 * d_head)
        k_out, v_out = sidecar(kv_window)

        assert k_out.shape == (batch_size, d_head)
        assert v_out.shape == (batch_size, d_head)

    def test_norm_preservation(self, sidecar):
        """Test that hard norm projection preserves scale."""
        batch_size = 4
        window_size = 8
        d_head = 64

        # Create input with specific norms
        k_input = torch.randn(batch_size, window_size, d_head) * 10
        v_input = torch.randn(batch_size, window_size, d_head) * 5
        kv_window = torch.cat([k_input, v_input], dim=-1)

        k_out, v_out = sidecar(kv_window)

        # Output norms should be close to input average norms
        k_input_norm = k_input.norm(dim=-1).mean(dim=1)
        v_input_norm = v_input.norm(dim=-1).mean(dim=1)
        k_out_norm = k_out.norm(dim=-1)
        v_out_norm = v_out.norm(dim=-1)

        # Should be approximately equal (hard projection)
        torch.testing.assert_close(k_out_norm, k_input_norm, rtol=0.1, atol=0.1)
        torch.testing.assert_close(v_out_norm, v_input_norm, rtol=0.1, atol=0.1)

    def test_gradient_flow(self, sidecar):
        """Test that gradients flow through the network."""
        batch_size = 2
        window_size = 8
        d_head = 64

        kv_window = torch.randn(batch_size, window_size, 2 * d_head, requires_grad=True)
        k_out, v_out = sidecar(kv_window)

        # Compute a scalar loss
        loss = k_out.sum() + v_out.sum()
        loss.backward()

        # Check gradients exist
        for param in sidecar.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSidecar:
    """Tests for Sidecar wrapper."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return LSTConfig(
            d_head=64,
            window_size=8,
            encoder_hidden_dim=128,
            encoder_num_layers=1,
        )

    def test_sidecar_wrapper(self, config):
        """Test Sidecar wrapper with separate K/V inputs."""
        sidecar = Sidecar(config)

        batch_size = 4
        keys = torch.randn(batch_size, config.window_size, config.d_head)
        values = torch.randn(batch_size, config.window_size, config.d_head)

        k_out, v_out = sidecar(keys, values)

        assert k_out.shape == (batch_size, config.d_head)
        assert v_out.shape == (batch_size, config.d_head)

    def test_num_parameters(self, config):
        """Test parameter counting."""
        sidecar = Sidecar(config)
        num_params = sidecar.num_parameters
        assert num_params > 0
        assert isinstance(num_params, int)


class TestBaselines:
    """Tests for baseline compression methods."""

    def test_h2o_import(self):
        """Test H2O baseline can be imported."""
        from src.baselines import H2O, H2OConfig

        config = H2OConfig(num_sink=4, num_recent=8)
        h2o = H2O(config)
        assert h2o.name == "H2O"

    def test_streaming_import(self):
        """Test StreamingLLM baseline can be imported."""
        from src.baselines import StreamingLLM, StreamingLLMConfig

        config = StreamingLLMConfig(num_sink=4, num_recent=8)
        streaming = StreamingLLM(config)
        assert streaming.name == "StreamingLLM"

    def test_tome_import(self):
        """Test ToMe baseline can be imported."""
        from src.baselines import ToMe, ToMeConfig

        config = ToMeConfig(num_sink=4, num_recent=8, r=4)
        tome = ToMe(config)
        assert tome.name == "ToMe"

    def test_h2o_compress(self):
        """Test H2O compression."""
        from src.baselines import H2O, H2OConfig

        config = H2OConfig(num_sink=4, num_recent=8, budget=20)
        h2o = H2O(config)

        batch_size = 2
        seq_len = 64
        d_head = 32

        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)

        k_out, v_out = h2o.compress(keys, values)

        # Should be reduced to budget
        assert k_out.shape[1] <= config.budget
        assert v_out.shape[1] <= config.budget

    def test_streaming_compress(self):
        """Test StreamingLLM compression."""
        from src.baselines import StreamingLLM, StreamingLLMConfig

        config = StreamingLLMConfig(num_sink=4, num_recent=8)
        streaming = StreamingLLM(config)

        batch_size = 2
        seq_len = 64
        d_head = 32

        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)

        k_out, v_out = streaming.compress(keys, values)

        # Should be reduced to sink + recent
        assert k_out.shape[1] == config.num_sink + config.num_recent
        assert v_out.shape[1] == config.num_sink + config.num_recent

    def test_kvmerger_import(self):
        """Test KVMerger baseline can be imported."""
        from src.baselines import KVMerger, KVMergerConfig

        config = KVMergerConfig(num_sink=4, num_recent=8)
        kvmerger = KVMerger(config)
        assert kvmerger.name == "KVMerger"

    def test_weightedkv_import(self):
        """Test WeightedKV baseline can be imported."""
        from src.baselines import WeightedKV, WeightedKVConfig

        config = WeightedKVConfig(num_sink=4, num_recent=8, budget=20)
        weightedkv = WeightedKV(config)
        assert weightedkv.name == "WeightedKV"

    def test_cam_import(self):
        """Test CaM baseline can be imported."""
        from src.baselines import CaM, CaMConfig

        config = CaMConfig(num_sink=4, num_recent=8, budget=20)
        cam = CaM(config)
        assert cam.name == "CaM"

    def test_tova_import(self):
        """Test TOVA baseline can be imported."""
        from src.baselines import TOVA, TOVAConfig

        config = TOVAConfig(num_sink=4, num_recent=8, budget=20)
        tova = TOVA(config)
        assert tova.name == "TOVA"

    def test_kvmerger_compress(self):
        """Test KVMerger compression with Gaussian kernel merging."""
        from src.baselines import KVMerger, KVMergerConfig

        config = KVMergerConfig(num_sink=4, num_recent=8, similarity_threshold=0.5)
        kvmerger = KVMerger(config)

        batch_size = 2
        seq_len = 64
        d_head = 32

        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)

        k_out, v_out = kvmerger.compress(keys, values)

        # Should reduce sequence length (by merging similar tokens)
        assert k_out.shape[0] == batch_size
        assert k_out.shape[2] == d_head
        assert k_out.shape[1] <= seq_len  # Merged, so should be reduced

    def test_weightedkv_compress(self):
        """Test WeightedKV compression with asymmetric K/V handling."""
        from src.baselines import WeightedKV, WeightedKVConfig

        config = WeightedKVConfig(num_sink=4, num_recent=8, budget=20)
        weightedkv = WeightedKV(config)

        batch_size = 2
        seq_len = 64
        d_head = 32

        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)

        k_out, v_out = weightedkv.compress(keys, values)

        # Should reduce to budget
        assert k_out.shape[1] <= config.budget
        assert v_out.shape[1] <= config.budget
        assert k_out.shape[0] == batch_size
        assert k_out.shape[2] == d_head

    def test_cam_compress(self):
        """Test CaM compression with value merging."""
        from src.baselines import CaM, CaMConfig

        config = CaMConfig(num_sink=4, num_recent=8, budget=20)
        cam = CaM(config)

        batch_size = 2
        seq_len = 64
        d_head = 32

        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)

        k_out, v_out = cam.compress(keys, values)

        # Should reduce to budget
        assert k_out.shape[1] <= config.budget
        assert v_out.shape[1] <= config.budget
        assert k_out.shape[0] == batch_size

    def test_tova_compress(self):
        """Test TOVA compression with greedy eviction."""
        from src.baselines import TOVA, TOVAConfig

        config = TOVAConfig(num_sink=4, num_recent=8, budget=20)
        tova = TOVA(config)

        batch_size = 2
        seq_len = 64
        d_head = 32

        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)

        k_out, v_out = tova.compress(keys, values)

        # Should reduce to budget
        assert k_out.shape[1] <= config.budget
        assert v_out.shape[1] <= config.budget
        assert k_out.shape[0] == batch_size

    def test_all_baselines_no_crash_on_short_seq(self):
        """Test that all baselines handle short sequences gracefully."""
        from src.baselines import (
            H2O, H2OConfig,
            StreamingLLM, StreamingLLMConfig,
            KVMerger, KVMergerConfig,
            WeightedKV, WeightedKVConfig,
            CaM, CaMConfig,
            TOVA, TOVAConfig,
        )

        batch_size = 2
        seq_len = 8  # Very short
        d_head = 32

        keys = torch.randn(batch_size, seq_len, d_head)
        values = torch.randn(batch_size, seq_len, d_head)

        # All methods should handle short sequences without crashing
        methods = [
            H2O(H2OConfig(num_sink=4, num_recent=4, budget=10)),
            StreamingLLM(StreamingLLMConfig(num_sink=4, num_recent=4)),
            KVMerger(KVMergerConfig(num_sink=2, num_recent=2)),
            WeightedKV(WeightedKVConfig(num_sink=2, num_recent=2, budget=10)),
            CaM(CaMConfig(num_sink=2, num_recent=2, budget=10)),
            TOVA(TOVAConfig(num_sink=2, num_recent=2, budget=10)),
        ]

        for method in methods:
            k_out, v_out = method.compress(keys, values)
            assert k_out.shape[0] == batch_size
            assert v_out.shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
