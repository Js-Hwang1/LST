"""
Force-Matched KV Cache Compression method.

Our novel method using learned coarse-graining with force matching.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseMethod, GenerationOutput, MethodConfig


class FMKVMethod(BaseMethod):
    """
    Force-Matched KV Cache Compression.
    
    Uses a trained Sidecar network to compress KV cache while
    preserving attention force dynamics.
    """
    
    def __init__(self, config: MethodConfig):
        super().__init__(config)
        self.sidecar = None
        self.compression_policy = None
        self._cache_stats = {}
    
    @property
    def name(self) -> str:
        return "fmkv"
    
    @property
    def is_compression_method(self) -> bool:
        return True
    
    def setup(self) -> None:
        """Load model, tokenizer, and Sidecar."""
        if self._is_setup:
            return
        
        print(f"[FMKV] Loading model: {self.config.model_name}")
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=self.config.torch_dtype_parsed,  # Use dtype instead of torch_dtype
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Load Sidecar if checkpoint provided
        sidecar_path = self.config.method_kwargs.get("sidecar_checkpoint")
        if sidecar_path:
            self._load_sidecar(sidecar_path)
        else:
            print("[FMKV] Warning: No Sidecar checkpoint provided, using untrained")
            self._init_default_sidecar()
        
        # Set up compression policy
        self._setup_compression_policy()
        
        self._is_setup = True
        print(f"[FMKV] Setup complete. Compression ratio target: {self.config.compression_ratio}")
    
    def _load_sidecar(self, checkpoint_path: str) -> None:
        """Load trained Sidecar from checkpoint."""
        from fmkv.sidecar import Sidecar, SidecarConfig
        
        print(f"[FMKV] Loading Sidecar from: {checkpoint_path}")
        # Set weights_only=False to allow custom classes (EncoderType, etc.)
        # This is safe since we're loading our own checkpoints
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.config.device,
            weights_only=False,
        )
        
        # Load config - prefer "sidecar_config" (new format) over "config" (old format)
        if "sidecar_config" in checkpoint:
            config_dict = checkpoint["sidecar_config"]
        elif "config" in checkpoint:
            # Old format - might have training config mixed in, filter to valid SidecarConfig fields
            config_dict = checkpoint["config"]
            # Filter to only SidecarConfig fields (exclude training config fields)
            from dataclasses import fields as dataclass_fields
            valid_fields = {f.name for f in dataclass_fields(SidecarConfig)}
            config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        else:
            raise ValueError("Checkpoint missing both 'sidecar_config' and 'config' keys")
        
        config = SidecarConfig(**config_dict)
        
        # Bug #5 Fix: Verify dimensions match model
        model_hidden_size = self.model.config.hidden_size
        model_num_heads = self.model.config.num_attention_heads
        model_head_dim = model_hidden_size // model_num_heads
        
        if config.d_head != model_head_dim:
            raise ValueError(
                f"Sidecar head dimension mismatch! "
                f"Sidecar expects d_head={config.d_head}, "
                f"but model has head_dim={model_head_dim} "
                f"(hidden_size={model_hidden_size} / num_heads={model_num_heads}). "
                f"Ensure Sidecar was trained on the same model architecture."
            )
        
        # Verify input/output dimensions
        expected_input_dim = 2 * model_head_dim  # Concatenated K and V
        if config.input_dim != expected_input_dim:
            raise ValueError(
                f"Sidecar input dimension mismatch! "
                f"Expected {expected_input_dim} (2 * {model_head_dim}), "
                f"got {config.input_dim}. "
                f"Sidecar should accept concatenated [K, V] vectors."
            )
        
        # Create and load Sidecar
        self.sidecar = Sidecar(config)
        self.sidecar.load_state_dict(checkpoint["model_state_dict"])
        self.sidecar.to(self.config.device)
        self.sidecar.eval()
        
        print(f"[FMKV] Sidecar loaded successfully:")
        print(f"  - Parameters: {sum(p.numel() for p in self.sidecar.parameters()):,}")
        print(f"  - Head dim: {config.d_head}")
        print(f"  - Input dim: {config.input_dim} (K+V concatenated)")
        print(f"  - Window size: {config.window_size}")
    
    def _init_default_sidecar(self) -> None:
        """Initialize default (untrained) Sidecar for testing."""
        from fmkv.sidecar import Sidecar, SidecarConfig
        
        # Get model hidden size
        hidden_size = self.model.config.hidden_size
        num_heads = self.model.config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        config = SidecarConfig(
            input_dim=head_dim,
            hidden_dim=256,
            output_dim=head_dim,
            num_encoder_layers=2,
            num_heads=4,
            encoder_type="transformer",
            aggregator_type="set_transformer",
        )
        
        self.sidecar = Sidecar(config)
        self.sidecar.to(self.config.device)
        self.sidecar.eval()
    
    def _setup_compression_policy(self) -> None:
        """Set up the compression policy."""
        from fmkv.compression.policy import WindowPolicy, BudgetPolicy
        
        # Bug #12 Fix: Enforce window_size consistency with training
        # The Sidecar was trained on a fixed window size (64 by default)
        # We must use the same window size during inference
        sidecar_window_size = self.sidecar.config.window_size if self.sidecar else 64
        
        # Determine compression policy from compression ratio or budget
        if self.config.compression_ratio is not None:
            # compression_ratio = fraction of tokens to KEEP (not compress away)
            # compression_ratio = 0.5 means keep 50% of tokens (compress away 50%)
            # compression_ratio = 0.1 means keep 10% of tokens (compress away 90%)
            
            # We MUST use the Sidecar's trained window size
            # To achieve different compression ratios, we adjust trigger frequency
            # not the window size itself
            window_size = sidecar_window_size
            
            # Calculate max_length (when to trigger compression)
            # Lower ratio = more aggressive compression = trigger earlier
            # After compression: cache_size = max_length - window_size + 1
            # We want: (max_length - window_size + 1) / max_length ≈ compression_ratio
            # Solving: max_length ≈ (window_size - 1) / (1 - compression_ratio)
            if self.config.compression_ratio > 0 and self.config.compression_ratio < 1:
                max_length = int((window_size - 1) / (1 - self.config.compression_ratio))
            else:
                max_length = 128
            
            # Clamp to reasonable range
            max_length = max(window_size + 10, min(max_length, 4096))
            
            print(f"[FMKV] Compression policy: window_size={window_size} (fixed from training), "
                  f"max_length={max_length} (trigger threshold)")
            
            self.compression_policy = WindowPolicy(
                window_size=window_size,
                max_length=max_length,
                min_dense_tokens=max(8, window_size // 4),  # Keep at least some dense tokens
            )
        elif self.config.cache_budget is not None:
            # Use budget policy for fixed cache size
            self.compression_policy = BudgetPolicy(
                budget_tokens=self.config.cache_budget,
                window_size=64,
            )
        else:
            # Default: window policy with standard settings
            self.compression_policy = WindowPolicy(
                window_size=64,
                max_length=128,
                min_dense_tokens=32,
            )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate with FMKV cache compression.
        
        This uses a modified generation loop that periodically
        compresses the KV cache using the Sidecar.
        """
        if not self._is_setup:
            self.setup()
        
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        
        original_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        
        # Track memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # For now, use standard generation with post-hoc compression tracking
        # Full integration requires modifying the generation loop
        # TODO: Implement custom generation with live compression
        
        with torch.no_grad():
            # Initial forward pass to get KV cache
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Track compression stats
            total_compressed = 0
            total_original = input_ids.shape[1]
            
            # Generate tokens one at a time
            generated_tokens = []
            
            for step in range(max_new_tokens):
                # Sample next token
                if kwargs.get("do_sample", False):
                    probs = torch.softmax(next_token_logits / kwargs.get("temperature", 1.0), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                generated_tokens.append(next_token)
                
                # Check for EOS
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
                
                # Compress KV cache if needed
                if self._should_compress(past_key_values, step):
                    past_key_values, num_compressed = self._compress_cache(past_key_values)
                    total_compressed += num_compressed
                
                # Forward pass with new token
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
            
            # Concatenate generated tokens
            if generated_tokens:
                generated = torch.cat(generated_tokens, dim=1)
                sequences = torch.cat([input_ids, generated], dim=1)
            else:
                sequences = input_ids
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Get memory stats
        peak_memory = 0.0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Compute final cache size
        final_cache_size = past_key_values[0][0].shape[2] if past_key_values else 0
        total_tokens = original_len + len(generated_tokens)
        compression_ratio = final_cache_size / total_tokens if total_tokens > 0 else 1.0
        
        # Store stats
        self._cache_stats = {
            "final_cache_size": final_cache_size,
            "total_tokens": total_tokens,
            "compressions_applied": total_compressed,
            "compression_ratio": compression_ratio,
        }
        
        # Decode output
        generated_text = self.tokenizer.batch_decode(
            sequences[:, original_len:],
            skip_special_tokens=True,
        )
        
        return GenerationOutput(
            sequences=sequences,
            text=generated_text,
            total_time=end_time - start_time,
            peak_memory_mb=peak_memory,
            cache_size=final_cache_size,
            original_size=original_len,
            compression_ratio=compression_ratio,
        )
    
    def _should_compress(self, past_key_values, step: int) -> bool:
        """Determine if compression should be applied."""
        if past_key_values is None or self.compression_policy is None:
            return False
        
        # Get current cache size
        cache_len = past_key_values[0][0].shape[2]
        
        # Use policy's should_compress method
        return self.compression_policy.should_compress(cache_len, new_tokens=1)
    
    def _compress_cache(
        self,
        past_key_values: tuple,
    ) -> tuple[tuple, int]:
        """
        Compress KV cache using Sidecar.
        
        Returns:
            Compressed past_key_values and number of tokens compressed
        """
        if self.sidecar is None:
            return past_key_values, 0
        
        window_size = self.compression_policy.window_size
        num_layers = len(past_key_values)
        
        compressed_kv = []
        total_compressed = 0
        
        for layer_idx in range(num_layers):
            keys, values = past_key_values[layer_idx]
            # keys/values: (batch, num_heads, seq_len, head_dim)
            
            batch_size, num_heads, seq_len, head_dim = keys.shape
            
            # Only compress if we have enough tokens
            if seq_len < window_size:
                compressed_kv.append((keys, values))
                continue
            
            # Compress oldest window
            # Keep recent tokens uncompressed for attention locality
            recent_len = min(window_size, seq_len // 2)
            compress_len = seq_len - recent_len
            
            if compress_len < window_size:
                compressed_kv.append((keys, values))
                continue
            
            # Split into windows for compression
            num_windows = compress_len // window_size
            
            compressed_keys_list = []
            compressed_values_list = []
            
            for w in range(num_windows):
                start_idx = w * window_size
                end_idx = start_idx + window_size
                
                # Extract window: (batch, num_heads, window_size, head_dim)
                k_window = keys[:, :, start_idx:end_idx, :]
                v_window = values[:, :, start_idx:end_idx, :]
                
                # Compress each head independently
                # Reshape for Sidecar: (batch * num_heads, window_size, head_dim)
                k_flat = k_window.reshape(-1, window_size, head_dim)
                v_flat = v_window.reshape(-1, window_size, head_dim)
                
                # Run through Sidecar - concatenate K and V first
                # Sidecar expects (batch, window_size, 2*d_head)
                kv_flat = torch.cat([k_flat, v_flat], dim=-1)  # (batch * num_heads, window_size, 2*head_dim)
                
                # Compress: (batch * num_heads, window_size, 2*head_dim) -> (batch * num_heads, 2*head_dim)
                kv_compressed = self.sidecar(kv_flat, return_split=False)  # (batch * num_heads, 2*head_dim)
                
                # Split back into K and V
                k_compressed, v_compressed = kv_compressed.chunk(2, dim=-1)  # Each: (batch * num_heads, head_dim)
                
                # Add sequence dimension: (batch * num_heads, head_dim) -> (batch * num_heads, 1, head_dim)
                k_compressed = k_compressed.unsqueeze(1)
                v_compressed = v_compressed.unsqueeze(1)
                
                # Reshape back: (batch, num_heads, 1, head_dim)
                k_compressed = k_compressed.view(batch_size, num_heads, -1, head_dim)
                v_compressed = v_compressed.view(batch_size, num_heads, -1, head_dim)
                
                compressed_keys_list.append(k_compressed)
                compressed_values_list.append(v_compressed)
                
                total_compressed += window_size - 1  # Each window -> 1 token
            
            # Concatenate compressed tokens with recent tokens
            if compressed_keys_list:
                compressed_keys = torch.cat(compressed_keys_list, dim=2)
                compressed_values = torch.cat(compressed_values_list, dim=2)
                
                # Add remaining uncompressed tokens
                remaining_start = num_windows * window_size
                if remaining_start < seq_len:
                    compressed_keys = torch.cat([
                        compressed_keys,
                        keys[:, :, remaining_start:, :]
                    ], dim=2)
                    compressed_values = torch.cat([
                        compressed_values,
                        values[:, :, remaining_start:, :]
                    ], dim=2)
                
                compressed_kv.append((compressed_keys, compressed_values))
            else:
                compressed_kv.append((keys, values))
        
        return tuple(compressed_kv), total_compressed
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss for perplexity evaluation.
        
        IMPORTANT: For perplexity evaluation, we use standard forward pass WITHOUT
        compression. This means perplexity scores will be identical to the dense
        baseline, and do NOT test compression effectiveness.
        
        To actually test compression, use the 'generation' benchmark instead, which
        applies compression during text generation (the intended use case).
        
        Rationale:
        - Perplexity measures log-likelihood on a fixed sequence
        - Compression is designed for autoregressive generation
        - Applying compression mid-sequence would be inconsistent with training
        - The theoretical framework preserves forces for FUTURE queries,
          not for evaluating a pre-existing sequence
        
        Returns:
            Per-token averaged loss (matching dense.py behavior).
        """
        if not self._is_setup:
            self.setup()
        
        input_ids = input_ids.to(self.model.device)
        labels = labels.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        
        with torch.no_grad():
            # Standard forward pass - same as dense method
            # The model handles label shifting and loss computation internally
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            
            # Validate loss is finite
            if not torch.isfinite(loss):
                import warnings
                warnings.warn(
                    f"Non-finite loss detected in FMKV compute_loss. "
                    f"input_ids shape: {input_ids.shape}, "
                    f"loss value: {loss.item()}"
                )
                # Return a large but finite loss instead of inf
                return torch.tensor(100.0, device=self.model.device)
        
        return loss
    
    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        stats = {
            "method": self.name,
            "is_compression": True,
            "compression_ratio": self.config.compression_ratio,
            "cache_budget": self.config.cache_budget,
        }
        stats.update(self._cache_stats)
        return stats
    
    def reset_cache(self) -> None:
        """Reset cache statistics."""
        self._cache_stats = {}

