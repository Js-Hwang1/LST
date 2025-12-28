#!/usr/bin/env python3
"""
LST Training Script
===================

Train LST sidecar networks for fixed-budget KV cache compression.

Usage:
    python scripts/train/train_lst.py \\
        --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --max_capacity_prompt 2048 \\
        --output_dir ./checkpoints/lst_2048

    # max_length is auto-computed to train with window sizes up to max_window_size
    # For longer sequences (more GPU memory):
    python scripts/train/train_lst.py \\
        --max_capacity_prompt 2048 \\
        --max_length 16384 \\
        --batch_size 1 \\
        --output_dir ./checkpoints/lst_2048

Training Objective:
    - L_ppl: Standard perplexity (generation quality)
    - L_qpaa: Query-probing attention alignment (query robustness)
    - L_div: Diversity (prevents collapse)
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.LST.sidecar import SidecarPPL
from src.LST.training import LSTTrainer, create_dataloaders
from src.LST.training.trainer import TrainerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_name: str, device: torch.device):
    """Load frozen language model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    # Get d_head (handle cases where head_dim exists but is None)
    d_head = getattr(model.config, "head_dim", None)
    if d_head is None:
        d_head = model.config.hidden_size // model.config.num_attention_heads

    logger.info(f"Model head dimension: {d_head}")

    return model, tokenizer, d_head


def main():
    parser = argparse.ArgumentParser(description="Train LST sidecar network")

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name",
    )

    # Sidecar architecture
    parser.add_argument("--hidden_dim", type=int, default=256, help="Sidecar hidden dim")
    parser.add_argument("--min_window_size", type=int, default=4, help="Minimum window size")
    parser.add_argument("--max_window_size", type=int, default=32, help="Maximum window size")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Encoder layers")

    # Training
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

    # Loss weights (NOVEL: QPAA)
    parser.add_argument("--lambda_ppl", type=float, default=1.0, help="PPL loss weight")
    parser.add_argument("--lambda_qpaa", type=float, default=0.5, help="QPAA loss weight")
    parser.add_argument("--lambda_diversity", type=float, default=0.1, help="Diversity loss weight")
    parser.add_argument("--num_probes", type=int, default=8, help="Number of query probes for QPAA")
    parser.add_argument(
        "--qpaa_warmup", type=int, default=200, help="Steps before QPAA is fully enabled"
    )

    # Compression
    parser.add_argument("--num_sink", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--num_recent", type=int, default=8, help="Number of recent tokens")
    parser.add_argument(
        "--max_capacity_prompt",
        type=int,
        required=True,
        help="Target KV cache budget (e.g., 512, 1024, 2048)",
    )

    # Data
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Sequence length (default: auto = max_capacity_prompt * 4, capped at 8192)",
    )
    parser.add_argument("--train_samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--val_samples", type=int, default=100, help="Validation samples")
    parser.add_argument(
        "--dataset",
        type=str,
        default="booksum",
        help="Dataset: 'booksum' (long), 'slimpajama', 'wikitext', or HuggingFace path",
    )
    parser.add_argument(
        "--cached_data",
        type=str,
        default=None,
        help="Path to pre-tokenized .pt file (skips tokenization)",
    )

    # Checkpointing
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/lst", help="Output directory"
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--log_steps", type=int, default=20, help="Log every N steps")

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Auto-compute max_length if not specified
    # Goal: train with window sizes up to max_window_size, but capped by memory
    if args.max_length is None:
        # Compute max_length that would produce window_size = max_window_size
        # But cap at 8192 for memory (longer needs gradient checkpointing)
        target_compressed = args.max_capacity_prompt - args.num_sink - args.num_recent
        ideal_max_length = args.max_window_size * target_compressed + args.num_sink + args.num_recent
        args.max_length = min(ideal_max_length, 8192)
        logger.info(f"Auto-computed max_length={args.max_length} (ideal={ideal_max_length})")

    # Validate
    if args.max_length <= args.max_capacity_prompt:
        parser.error(
            f"max_length ({args.max_length}) must be > max_capacity_prompt ({args.max_capacity_prompt})"
        )

    # Compute and log training window size
    target_compressed = args.max_capacity_prompt - args.num_sink - args.num_recent
    middle_tokens = args.max_length - args.num_sink - args.num_recent
    training_window_size = (middle_tokens + target_compressed - 1) // target_compressed
    logger.info(f"Training window size: {training_window_size} (max_window_size={args.max_window_size})")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, tokenizer, d_head = load_model(args.model_name, device)

    # Create sidecar (match model dtype for CUDA)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    sidecar = SidecarPPL(
        d_head=d_head,
        max_window_size=args.max_window_size,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
    ).to(device=device, dtype=model_dtype)

    num_params = sum(p.numel() for p in sidecar.parameters())
    logger.info(f"Sidecar parameters: {num_params:,}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
        cached_path=args.cached_data,
        dataset_name=args.dataset,
    )

    # Create trainer config
    config = TrainerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lambda_ppl=args.lambda_ppl,
        lambda_qpaa=args.lambda_qpaa,
        lambda_diversity=args.lambda_diversity,
        qpaa_warmup_steps=args.qpaa_warmup,
        num_probes=args.num_probes,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        max_capacity_prompt=args.max_capacity_prompt,
        num_sink=args.num_sink,
        num_recent=args.num_recent,
        log_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    logger.info(f"Training for target: max_capacity_prompt={args.max_capacity_prompt}")

    # Create trainer
    trainer = LSTTrainer(
        model=model,
        sidecar=sidecar,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
