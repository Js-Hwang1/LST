#!/usr/bin/env python3
"""
Data Collector for LST Training
================================

Pre-tokenizes WikiText-103 samples and saves them for fast loading.
This avoids the 5+ minute tokenization wait on each training run.

Usage:
    python scripts/collect_data.py --output data/wikitext_512.pt --num_samples 10000

The output file can be committed to the repo or uploaded separately.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def collect_samples(
    tokenizer,
    max_length: int = 512,
    num_samples: int = 10000,
    split: str = "train",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-v1",
    seed: int = 42,
    min_length: int = 100,
) -> list[torch.Tensor]:
    """
    Collect and tokenize samples from WikiText.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_samples: Number of samples to collect
        split: Dataset split
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        seed: Random seed
        min_length: Minimum text length in characters

    Returns:
        List of tokenized samples (each is a tensor of shape (max_length,))
    """
    from datasets import load_dataset

    logger.info(f"Loading {dataset_name}/{dataset_config} ({split})...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    dataset = dataset.shuffle(seed=seed)

    samples = []
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    logger.info(f"Tokenizing and filtering for samples close to {max_length} tokens...")
    for item in tqdm(dataset, desc=f"Collecting {split}", total=len(dataset)):
        text = item.get("text", "")
        if len(text.strip()) < min_length:
            continue

        # Tokenize without padding to check actual length
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        actual_length = tokens["input_ids"].shape[1]

        # Only keep samples close to max_length (within 10 tokens)
        if actual_length >= max_length - 10:
            # Pad to exact max_length
            if actual_length < max_length:
                pad_length = max_length - actual_length
                padding = torch.full((1, pad_length), pad_token_id, dtype=tokens["input_ids"].dtype)
                tokens["input_ids"] = torch.cat([tokens["input_ids"], padding], dim=1)
            samples.append(tokens["input_ids"].squeeze(0))

        if len(samples) >= num_samples:
            break

    logger.info(f"Collected {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Collect and cache tokenized data for LST training")

    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name (for tokenizer)",
    )
    parser.add_argument("--output", type=str, default="data/wikitext_512.pt", help="Output file path")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--train_samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=500, help="Number of validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect training samples
    logger.info("Collecting training samples...")
    train_samples = collect_samples(
        tokenizer,
        max_length=args.max_length,
        num_samples=args.train_samples,
        split="train",
        seed=args.seed,
    )

    # Collect validation samples
    logger.info("Collecting validation samples...")
    val_samples = collect_samples(
        tokenizer,
        max_length=args.max_length,
        num_samples=args.val_samples,
        split="validation",
        seed=args.seed + 1,
    )

    # Stack into tensors
    train_tensor = torch.stack(train_samples)
    val_tensor = torch.stack(val_samples)

    # Save
    data = {
        "train": train_tensor,
        "val": val_tensor,
        "max_length": args.max_length,
        "model_name": args.model_name,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }

    torch.save(data, output_path)

    # Get file size
    file_size = output_path.stat().st_size / (1024 * 1024)

    logger.info(f"Saved to {output_path}")
    logger.info(f"  Train samples: {len(train_samples)}")
    logger.info(f"  Val samples: {len(val_samples)}")
    logger.info(f"  File size: {file_size:.2f} MB")
    logger.info(f"  Shape: train={train_tensor.shape}, val={val_tensor.shape}")


if __name__ == "__main__":
    main()
