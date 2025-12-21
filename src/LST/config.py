"""
LST Configuration
=================

Configuration dataclasses for the LST compression system.
Designed for reproducibility and easy hyperparameter tuning.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from enum import Enum


class EncoderType(str, Enum):
    """Encoder architecture type."""

    TRANSFORMER = "transformer"
    GIN = "gin"  # Graph Isomorphism Network
    MLP = "mlp"  # Simple MLP baseline


class AggregatorType(str, Enum):
    """Aggregation method for N -> 1 compression."""

    SET_TRANSFORMER = "set_transformer"  # Learned attention pooling
    MEAN_POOL = "mean_pool"  # Simple averaging (baseline)
    ATTENTION_POOL = "attention_pool"  # Single-head attention pooling


@dataclass
class LSTConfig:
    """
    Configuration for the LST sidecar network.

    The sidecar maps a window of N KV pairs to a single super-token
    that preserves language modeling quality.

    Attributes:
        d_head: Dimension of each attention head in the base model.
        n_heads: Number of attention heads in the base model.
        window_size: Number of tokens (N) in each compression window.
        num_windows: Number of windows per sample (for positional embeddings).
        encoder_type: Type of encoder architecture.
        encoder_hidden_dim: Hidden dimension of the encoder.
        encoder_num_layers: Number of encoder layers.
        encoder_num_heads: Number of attention heads in encoder.
        encoder_dropout: Dropout rate in encoder.
        aggregator_type: Type of aggregation method.
        aggregator_num_heads: Number of heads in aggregator.
        aggregator_num_inducing: Number of inducing points (Set Transformer).
        use_layer_norm: Whether to use layer normalization.
        position_encoding: Type of positional encoding.
        output_projection: Whether to use output projection layer.
    """

    # Base model dimensions (set at runtime based on target LLM)
    d_head: int = 128
    n_heads: int = 32

    # Compression window
    window_size: int = 8
    num_windows: int = 4  # For window positional embeddings

    # Encoder configuration
    encoder_type: EncoderType = EncoderType.TRANSFORMER
    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 2
    encoder_num_heads: int = 4
    encoder_dropout: float = 0.0  # CRITICAL: Use 0.0 for eval stability
    encoder_ffn_ratio: float = 4.0

    # Aggregator configuration
    aggregator_type: AggregatorType = AggregatorType.ATTENTION_POOL
    aggregator_num_heads: int = 4
    aggregator_num_inducing: int = 8

    # Architecture options
    use_layer_norm: bool = True
    use_residual: bool = True
    position_encoding: Literal["sinusoidal", "learned", "rope", "none"] = "learned"
    output_projection: bool = True

    # Hard norm projection (prevents collapse/explosion)
    use_hard_norm: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        assert self.d_head > 0, "d_head must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.window_size >= 2, "window_size must be at least 2"
        assert self.encoder_num_layers >= 1, "encoder_num_layers must be at least 1"

        # Convert string enums if needed
        if isinstance(self.encoder_type, str):
            self.encoder_type = EncoderType(self.encoder_type)
        if isinstance(self.aggregator_type, str):
            self.aggregator_type = AggregatorType(self.aggregator_type)

    @property
    def input_dim(self) -> int:
        """Input dimension: concatenated K and V vectors."""
        return 2 * self.d_head

    @property
    def output_dim(self) -> int:
        """Output dimension: K_CG and V_CG concatenated."""
        return 2 * self.d_head

    @classmethod
    def from_model(
        cls,
        model_name_or_config,
        window_size: int = 8,
        **kwargs,
    ) -> "LSTConfig":
        """
        Create LSTConfig from a HuggingFace model name or config.

        Args:
            model_name_or_config: HuggingFace model name or config object.
            window_size: Compression window size.
            **kwargs: Additional config overrides.

        Returns:
            LSTConfig configured for the target model.
        """
        from transformers import AutoConfig

        if isinstance(model_name_or_config, str):
            hf_config = AutoConfig.from_pretrained(model_name_or_config)
        else:
            hf_config = model_name_or_config

        # Extract dimensions from various model architectures
        if hasattr(hf_config, "head_dim"):
            d_head = hf_config.head_dim
        elif hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_attention_heads"):
            d_head = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise ValueError(f"Cannot infer head dimension from config: {type(hf_config)}")

        n_heads = getattr(hf_config, "num_attention_heads", 32)

        return cls(
            d_head=d_head,
            n_heads=n_heads,
            window_size=window_size,
            **kwargs,
        )


@dataclass
class TrainingConfig:
    """Configuration for LST training."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 2000

    # Batch size
    batch_size: int = 512
    gradient_accumulation_steps: int = 1

    # Regularization
    gradient_clip_norm: float = 1.0

    # Compression parameters
    num_sink: int = 4
    num_recent: int = 8
    window_size: int = 8
    num_windows_per_sample: int = 4

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 200

    # Logging
    log_steps: int = 20
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Seeds
    seed: int = 42
    val_seed: int = 123

    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    trajectories_path: str = "./data/trajectories"
    output_dir: str = "./checkpoints"

    def __post_init__(self) -> None:
        """Validate training configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_steps > 0, "max_steps must be positive"
