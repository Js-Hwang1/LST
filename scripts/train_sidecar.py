#!/usr/bin/env python3
"""
Train Sidecar Network
=====================

Main script for training the Sidecar network on collected trajectories.

Usage:
    python scripts/train_sidecar.py \
        --model_name meta-llama/Llama-2-7b-hf \
        --trajectories_path ./data/trajectories \
        --output_dir ./checkpoints \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --max_steps 100000

Or with config file:
    python scripts/train_sidecar.py --config configs/training/default.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig

from fmkv.sidecar import Sidecar, SidecarConfig
from fmkv.training import SidecarTrainer, TrainingConfig
from fmkv.data import ForceMatchingDataset, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sidecar for KV cache compression")
    
    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name (for extracting dimensions)",
    )
    
    # Data
    parser.add_argument(
        "--trajectories_path",
        type=str,
        required=True,
        help="Path to collected trajectories",
    )
    parser.add_argument(
        "--gradients_path",
        type=str,
        default=None,
        help="Path to precomputed gradients (optional)",
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.1,
        help="Fraction of data for evaluation",
    )
    
    # Sidecar architecture
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Compression window size",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="transformer",
        choices=["transformer", "gin", "mlp"],
        help="Encoder architecture",
    )
    parser.add_argument(
        "--encoder_hidden_dim",
        type=int,
        default=256,
        help="Encoder hidden dimension",
    )
    parser.add_argument(
        "--encoder_num_layers",
        type=int,
        default=3,
        help="Number of encoder layers",
    )
    
    # Bug #19 Fix: Multi-window training
    parser.add_argument(
        "--num_windows_per_sample",
        type=int,
        default=4,
        help="Number of windows per training sample (for multi-window training)",
    )
    parser.add_argument(
        "--use_multi_window",
        action="store_true",
        help="Use multi-window dataset (trains with multiple super-tokens)",
    )
    
    # Training
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="fmkv_sidecar",
        help="Experiment name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Warmup steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    
    # Loss weights
    parser.add_argument(
        "--force_matching_weight",
        type=float,
        default=1.0,
        help="Weight for force matching loss",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.1,
        help="Weight for consistency loss",
    )
    
    # Precision
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Training dtype",
    )
    
    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="fmkv",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Get model dimensions
    print(f"Loading model config from {args.model_name}...")
    try:
        hf_config = AutoConfig.from_pretrained(args.model_name)
        
        # Extract head dimension
        if hasattr(hf_config, "head_dim"):
            d_head = hf_config.head_dim
        else:
            d_head = hf_config.hidden_size // hf_config.num_attention_heads
        
        n_heads = hf_config.num_attention_heads
    except Exception as e:
        print(f"Warning: Could not load model config: {e}")
        print("Using default dimensions (d_head=128, n_heads=32)")
        d_head = 128
        n_heads = 32
    
    print(f"Model dimensions: d_head={d_head}, n_heads={n_heads}")
    
    # Create Sidecar config
    sidecar_config = SidecarConfig(
        d_head=d_head,
        n_heads=n_heads,
        window_size=args.window_size,
        encoder_type=args.encoder_type,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_num_layers=args.encoder_num_layers,
        dtype=args.dtype,
    )
    
    print(f"\nSidecar config:")
    print(f"  Window size: {sidecar_config.window_size}")
    print(f"  Encoder: {sidecar_config.encoder_type}")
    print(f"  Hidden dim: {sidecar_config.encoder_hidden_dim}")
    print(f"  Layers: {sidecar_config.encoder_num_layers}")
    print(f"  Estimated params: {sidecar_config.estimate_parameters():,}")
    
    # Create Sidecar
    sidecar = Sidecar(sidecar_config)
    print(f"  Actual params: {sidecar.num_parameters:,}")
    
    # Load data
    print(f"\nLoading trajectories from {args.trajectories_path}...")
    
    if args.use_multi_window:
        # Bug #19 Fix: Use multi-window dataset for non-trivial attention
        from fmkv.data.multi_window_dataset import MultiWindowDataset
        
        dataset = MultiWindowDataset.from_trajectories(
            trajectories_path=args.trajectories_path,
            num_windows_per_sample=args.num_windows_per_sample,
            window_size=args.window_size,
            d_head=d_head,
        )
        print(f"Using multi-window dataset: {args.num_windows_per_sample} windows per sample")
    else:
        # Standard single-window dataset
        dataset = ForceMatchingDataset.from_trajectories(
            trajectories_path=args.trajectories_path,
            gradients_path=args.gradients_path,
            window_size=args.window_size,
            d_head=d_head,
        )
    
    print(f"Loaded {len(dataset)} training samples")
    
    # Split into train/eval
    eval_size = int(len(dataset) * args.eval_split)
    train_size = len(dataset) - eval_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    eval_dataloader = create_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Create training config
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        force_matching_weight=args.force_matching_weight,
        consistency_weight=args.consistency_weight,
        mixed_precision=args.mixed_precision,
        dtype=args.dtype,
        log_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        num_workers=args.num_workers,
    )
    
    # Create trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    trainer = SidecarTrainer(
        sidecar=sidecar,
        config=training_config,
        device=device,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        resume_from=args.resume_from,
    )
    
    print("\nTraining complete!")
    print(f"Checkpoints saved to: {training_config.output_dir}")


if __name__ == "__main__":
    main()

