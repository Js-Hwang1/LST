"""
Training Datasets for LST
=========================

Efficient text datasets for training LST sidecars.

Supports two modes:
1. On-the-fly tokenization (slow, ~5 minutes)
2. Pre-cached tokenization (fast, <1 second)

To create cached data, run:
    python scripts/collection/collect_data.py --output data/wikitext_512.pt
"""

import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Pre-tokenized text dataset for efficient training.

    Loads and tokenizes text samples, filtering for appropriate length.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_samples: Number of samples to load
        split: Dataset split ('train', 'validation', 'test')
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        num_samples: int = 10000,
        split: str = "train",
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-v1",
        seed: int = 42,
        min_length: int = 100,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples

        from datasets import load_dataset

        dataset = load_dataset(dataset_name, dataset_config, split=split)
        dataset = dataset.shuffle(seed=seed)

        # Pre-tokenize samples
        self.samples = []
        for item in tqdm(
            dataset,
            desc=f"Tokenizing {split}",
            total=min(num_samples * 2, len(dataset)),
        ):
            text = item.get("text", "")
            if len(text.strip()) < min_length:
                continue

            # First tokenize without padding to check actual length
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            actual_length = tokens["input_ids"].shape[1]

            # Only keep samples that are close to max_length (within 10 tokens)
            if actual_length >= max_length - 10:
                # Pad to exact max_length for consistent batch sizes
                if actual_length < max_length:
                    pad_length = max_length - actual_length
                    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                    padding = torch.full(
                        (1, pad_length), pad_token_id, dtype=tokens["input_ids"].dtype
                    )
                    tokens["input_ids"] = torch.cat([tokens["input_ids"], padding], dim=1)
                self.samples.append(tokens["input_ids"].squeeze(0))

            if len(self.samples) >= num_samples:
                break

        logger.info(f"Loaded {len(self.samples)} samples for {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tensor:
        return self.samples[idx]


class CachedTextDataset(Dataset):
    """
    Dataset that loads pre-tokenized samples from a cached .pt file.

    Much faster than TextDataset (~1 second vs ~5 minutes).

    Args:
        cached_path: Path to cached .pt file
        split: 'train' or 'val'
    """

    def __init__(self, cached_path: str | Path, split: str = "train"):
        cached_path = Path(cached_path)
        if not cached_path.exists():
            raise FileNotFoundError(
                f"Cached data not found at {cached_path}. "
                f"Run: python scripts/collection/collect_data.py --output {cached_path}"
            )

        logger.info(f"Loading cached data from {cached_path}")
        data = torch.load(cached_path, weights_only=True)

        key = "train" if split == "train" else "val"
        self.samples = data[key]
        self.max_length = data.get("max_length", self.samples.shape[1])

        logger.info(f"Loaded {len(self.samples)} {split} samples (max_length={self.max_length})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tensor:
        return self.samples[idx]


def collate_fn(batch):
    """Stack batch of tensors."""
    return torch.stack(batch)


def create_dataloaders(
    tokenizer=None,
    batch_size: int = 4,
    max_length: int = 512,
    train_samples: int = 5000,
    val_samples: int = 100,
    seed: int = 42,
    num_workers: int = 0,
    cached_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        tokenizer: HuggingFace tokenizer (not needed if using cached_path)
        batch_size: Batch size
        max_length: Maximum sequence length
        train_samples: Number of training samples
        val_samples: Number of validation samples
        seed: Random seed
        num_workers: Number of dataloader workers
        cached_path: Path to pre-tokenized .pt file (fast mode)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if cached_path is not None:
        # Fast mode: load from pre-cached file
        train_dataset = CachedTextDataset(cached_path, split="train")
        val_dataset = CachedTextDataset(cached_path, split="val")
    else:
        # Slow mode: tokenize on-the-fly
        if tokenizer is None:
            raise ValueError("tokenizer is required when cached_path is not provided")

        train_dataset = TextDataset(
            tokenizer,
            max_length=max_length,
            num_samples=train_samples,
            split="train",
            seed=seed,
        )

        val_dataset = TextDataset(
            tokenizer,
            max_length=max_length,
            num_samples=val_samples,
            split="validation",
            seed=seed + 1,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
