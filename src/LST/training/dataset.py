"""
Training Datasets for LST
=========================

Efficient text datasets for training LST sidecars.
"""

import logging

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

        dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
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

            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Only keep samples that are close to max_length
            if tokens["input_ids"].shape[1] >= max_length - 10:
                self.samples.append(tokens["input_ids"].squeeze(0))

            if len(self.samples) >= num_samples:
                break

        logger.info(f"Loaded {len(self.samples)} samples for {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tensor:
        return self.samples[idx]


def collate_fn(batch):
    """Stack batch of tensors."""
    return torch.stack(batch)


def create_dataloaders(
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    train_samples: int = 5000,
    val_samples: int = 100,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        train_samples: Number of training samples
        val_samples: Number of validation samples
        seed: Random seed
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
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
