"""
Training Datasets for LST
=========================

Efficient text datasets for training LST sidecars.

Key insight: Training sequences MUST be longer than max_capacity_prompt
for compression to occur.

Recommended datasets for long-context training:
- booksum: Book chapters, avg ~6700 tokens (best for 2K+ budgets)
- slimpajama: Mixed web data, up to ~5K tokens
- wikitext: Short paragraphs, uses concatenation mode

Supports:
1. Naturally long datasets (booksum) - filters for docs >= max_length
2. Concatenation mode for short datasets (wikitext)
3. Pre-cached tokenization for fast loading
"""

import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Dataset configurations for long-context training
DATASET_CONFIGS = {
    "booksum": {
        "name": "kmfoda/booksum",
        "config": None,
        "text_field": "chapter",
        "avg_tokens": 6700,
        "val_split": "validation",
    },
    "slimpajama": {
        "name": "cerebras/SlimPajama-627B",
        "config": None,
        "text_field": "text",
        "avg_tokens": 650,
        "val_split": "validation",
    },
    "wikitext": {
        "name": "wikitext",
        "config": "wikitext-103-v1",
        "text_field": "text",
        "avg_tokens": 100,
        "val_split": "validation",
    },
}


class LongTextDataset(Dataset):
    """
    Dataset for long-context training.

    For naturally long datasets (booksum), filters for documents >= max_length.
    For short datasets (wikitext), concatenates documents to reach max_length.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Target sequence length (should be > max_capacity_prompt)
        num_samples: Number of samples to create
        split: Dataset split ('train', 'validation')
        dataset_name: Dataset key ('booksum', 'slimpajama', 'wikitext') or HF path
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 4096,
        num_samples: int = 5000,
        split: str = "train",
        dataset_name: str = "booksum",
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples

        from datasets import load_dataset

        # Resolve dataset config
        if dataset_name in DATASET_CONFIGS:
            cfg = DATASET_CONFIGS[dataset_name]
            hf_name = cfg["name"]
            hf_config = cfg["config"]
            text_field = cfg["text_field"]
            avg_tokens = cfg["avg_tokens"]
            # Use correct split name for validation
            if split == "validation":
                split = cfg.get("val_split", "validation")
        else:
            # Assume it's a direct HuggingFace path
            hf_name = dataset_name
            hf_config = None
            text_field = "text"
            avg_tokens = 500  # Unknown, assume moderate

        logger.info(f"Loading dataset: {hf_name} (split={split})")

        if hf_config:
            dataset = load_dataset(hf_name, hf_config, split=split)
        else:
            dataset = load_dataset(hf_name, split=split)
        dataset = dataset.shuffle(seed=seed)

        # Decide strategy based on expected document length
        if avg_tokens >= max_length * 0.5:
            # Naturally long - filter for documents >= max_length
            self._load_long_documents(dataset, text_field, tokenizer, max_length, num_samples)
        else:
            # Short documents - concatenate
            self._load_concatenated(dataset, text_field, tokenizer, max_length, num_samples)

    def _load_long_documents(self, dataset, text_field, tokenizer, max_length, num_samples):
        """Load naturally long documents, truncating to max_length."""
        logger.info(f"Filtering for documents with >= {max_length} tokens...")

        self.samples = []
        skipped = 0

        for doc in tqdm(dataset, desc="Filtering", total=min(num_samples * 3, len(dataset))):
            text = doc.get(text_field, "")
            if not text or not text.strip():
                continue

            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"].squeeze(0)

            if len(tokens) >= max_length:
                self.samples.append(tokens[:max_length])
            else:
                skipped += 1

            if len(self.samples) >= num_samples:
                break

        if len(self.samples) < num_samples:
            logger.warning(
                f"Only found {len(self.samples)} documents >= {max_length} tokens. "
                f"Consider using a shorter max_length or different dataset."
            )

        logger.info(f"Created {len(self.samples)} samples (skipped {skipped} short docs)")

    def _load_concatenated(self, dataset, text_field, tokenizer, max_length, num_samples):
        """Concatenate short documents to create long sequences."""
        logger.info(f"Concatenating documents to length {max_length}...")

        all_text = " ".join(
            doc[text_field].strip()
            for doc in dataset
            if doc.get(text_field) and doc[text_field].strip()
        )

        logger.info(f"Tokenizing concatenated text (~{len(all_text)//1000}K chars)...")
        all_tokens = tokenizer(
            all_text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )["input_ids"].squeeze(0)

        total_tokens = len(all_tokens)
        logger.info(f"Total tokens: {total_tokens:,}")

        self.samples = []
        for start in range(0, total_tokens - max_length, max_length):
            if len(self.samples) >= num_samples:
                break
            self.samples.append(all_tokens[start : start + max_length])

        if len(self.samples) < num_samples:
            logger.warning(
                f"Only created {len(self.samples)} samples (requested {num_samples}). "
                f"Consider using a larger dataset."
            )

        logger.info(f"Created {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tensor:
        return self.samples[idx]


class CachedTextDataset(Dataset):
    """
    Dataset that loads pre-tokenized samples from a cached .pt file.

    Much faster than on-the-fly tokenization.

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
    max_length: int = 4096,
    train_samples: int = 5000,
    val_samples: int = 100,
    seed: int = 42,
    num_workers: int = 0,
    cached_path: str | Path | None = None,
    dataset_name: str = "booksum",
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders with long sequences.

    Args:
        tokenizer: HuggingFace tokenizer (not needed if using cached_path)
        batch_size: Batch size
        max_length: Sequence length (MUST be > max_capacity_prompt for compression)
        train_samples: Number of training samples
        val_samples: Number of validation samples
        seed: Random seed
        num_workers: Number of dataloader workers
        cached_path: Path to pre-tokenized .pt file (fast mode)
        dataset_name: Dataset key ('booksum', 'slimpajama', 'wikitext') or HF path

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if cached_path is not None:
        train_dataset = CachedTextDataset(cached_path, split="train")
        val_dataset = CachedTextDataset(cached_path, split="val")
    else:
        if tokenizer is None:
            raise ValueError("tokenizer is required when cached_path is not provided")

        train_dataset = LongTextDataset(
            tokenizer,
            max_length=max_length,
            num_samples=train_samples,
            split="train",
            dataset_name=dataset_name,
            seed=seed,
        )

        val_dataset = LongTextDataset(
            tokenizer,
            max_length=max_length,
            num_samples=val_samples,
            split="validation",
            dataset_name=dataset_name,
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
