#!/usr/bin/env python3
"""
LST Training Script
===================

Lightweight wrapper for training LST sidecar networks.

Usage:
    python scripts/train/train_lst.py \\
        --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --max_steps 2000 \\
        --output_dir ./checkpoints/tinyllama_lst

For Llama-2-7B:
    python scripts/train/train_lst.py \\
        --model_name meta-llama/Llama-2-7b-hf \\
        --max_steps 2000 \\
        --batch_size 256 \\
        --output_dir ./checkpoints/llama2_lst

Novel Training Objective (QPAA):
    The training uses Query-Probing Attention Alignment (QPAA) in addition
    to standard PPL loss. This ensures super-tokens work for arbitrary
    future queries, not just training continuations.

Loss Components:
    - L_ppl: Standard perplexity (generation quality)
    - L_qpaa: Random query probing (query robustness) [NOVEL]
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    # Get d_head
    if hasattr(model.config, "head_dim"):
        d_head = model.config.head_dim
    else:
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
    parser.add_argument("--window_size", type=int, default=8, help="Compression window size")
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
    parser.add_argument("--qpaa_warmup", type=int, default=200, help="Steps before QPAA is fully enabled")

    # Compression
    parser.add_argument("--num_sink", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--num_recent", type=int, default=8, help="Number of recent tokens")

    # Data
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--train_samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--val_samples", type=int, default=100, help="Validation samples")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./checkpoints/lst", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--log_steps", type=int, default=20, help="Log every N steps")

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, tokenizer, d_head = load_model(args.model_name, device)

    # Create sidecar
    sidecar = SidecarPPL(
        d_head=d_head,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
    ).to(device)

    num_params = sum(p.numel() for p in sidecar.parameters())
    logger.info(f"Sidecar parameters: {num_params:,}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
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
        window_size=args.window_size,
        num_sink=args.num_sink,
        num_recent=args.num_recent,
        log_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        seed=args.seed,
    )

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
