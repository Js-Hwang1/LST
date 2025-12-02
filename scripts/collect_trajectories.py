#!/usr/bin/env python3
"""
Trajectory Collection Script
============================

Collects KV cache trajectories from a frozen LLM for training the Sidecar.

This script:
1. Loads a pretrained LLM
2. Runs forward passes on text data
3. Extracts KV states at each layer
4. Samples future queries for each window
5. Saves trajectories to disk for offline training

Usage:
    python scripts/collect_trajectories.py \
        --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --dataset_name wikitext \
        --dataset_config wikitext-103-raw-v1 \
        --output_dir ./data/trajectories \
        --num_samples 10000 \
        --window_size 64 \
        --batch_size 4

For larger models:
    python scripts/collect_trajectories.py \
        --model_name meta-llama/Llama-2-7b-hf \
        --dataset_name allenai/c4 \
        --dataset_config en \
        --output_dir ./data/trajectories_llama2 \
        --num_samples 50000 \
        --batch_size 2 \
        --torch_dtype bfloat16
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# HuggingFace imports
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmkv.sidecar import SidecarConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect KV cache trajectories for Sidecar training"
    )
    
    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for custom models",
    )
    
    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-103-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column containing text data",
    )
    
    # Trajectory collection
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/trajectories",
        help="Output directory for trajectories",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of text samples to process",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--min_seq_length",
        type=int,
        default=512,
        help="Minimum sequence length (skip shorter)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing",
    )
    
    # Window configuration
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Size of KV windows to extract",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Stride between windows",
    )
    parser.add_argument(
        "--num_future_queries",
        type=int,
        default=16,
        help="Number of future queries per window",
    )
    parser.add_argument(
        "--future_query_range",
        type=int,
        default=128,
        help="Range to sample future queries from",
    )
    
    # Layer selection
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g., '0,1,2'). None = all layers",
    )
    parser.add_argument(
        "--sample_layers",
        type=int,
        default=None,
        help="Randomly sample N layers per batch (for efficiency)",
    )
    
    # Checkpointing
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save checkpoint every N windows",
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
        help="DataLoader workers",
    )
    
    return parser.parse_args()


class TextDataset(Dataset):
    """Simple dataset wrapper for text data."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 2048,
        min_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        
        # Filter and tokenize
        self.samples = []
        for text in tqdm(texts, desc="Tokenizing"):
            if not text or len(text.strip()) < 100:
                continue
            
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                padding=False,
            )
            
            if tokens["input_ids"].size(1) >= min_length:
                self.samples.append({
                    "input_ids": tokens["input_ids"].squeeze(0),
                    "attention_mask": tokens["attention_mask"].squeeze(0),
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id: int = 0):
    """Collate function for variable-length sequences."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@torch.no_grad()
def extract_kv_from_model(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layers: Optional[List[int]] = None,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract KV states from model forward pass.
    
    Returns:
        Dict mapping layer_idx -> (keys, values)
        Keys/Values shape: (batch, num_heads, seq_len, head_dim)
    """
    # Forward pass with cache
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    
    # Extract KV cache
    past_key_values = outputs.past_key_values
    hidden_states = outputs.hidden_states
    
    kv_states = {}
    
    num_layers = len(past_key_values)
    if target_layers is None:
        target_layers = list(range(num_layers))
    
    for layer_idx in target_layers:
        if layer_idx >= num_layers:
            continue
        
        # past_key_values[layer] is tuple of (key, value)
        # Shape: (batch, num_heads, seq_len, head_dim)
        keys, values = past_key_values[layer_idx]
        kv_states[layer_idx] = (keys.cpu(), values.cpu())
    
    # Also return hidden states for query projection
    hidden_dict = {
        i: h.cpu() for i, h in enumerate(hidden_states)
        if target_layers is None or i in target_layers
    }
    
    return kv_states, hidden_dict


def extract_windows(
    kv_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    hidden_states: Dict[int, torch.Tensor],
    window_size: int,
    stride: int,
    num_future_queries: int,
    future_query_range: int,
    attention_mask: torch.Tensor,
) -> List[Dict]:
    """
    Extract sliding windows from KV states.
    
    Returns list of window dicts with:
        - layer_idx
        - keys: (batch, window_size, d_head) - using first head
        - values: (batch, window_size, d_head)
        - queries: (batch, num_queries, hidden_dim)
        - window_start
    """
    windows = []
    
    batch_size = attention_mask.size(0)
    seq_lengths = attention_mask.sum(dim=1).tolist()
    
    for layer_idx, (keys, values) in kv_states.items():
        # keys shape: (batch, num_heads, seq_len, head_dim)
        _, num_heads, seq_len, head_dim = keys.shape
        
        # Get hidden states for this layer
        hidden = hidden_states.get(layer_idx)
        if hidden is None:
            hidden = hidden_states.get(layer_idx + 1)  # Sometimes off by one
        
        for batch_idx in range(batch_size):
            actual_len = seq_lengths[batch_idx]
            
            # Calculate number of windows for this sequence
            num_windows = (actual_len - window_size - future_query_range) // stride
            
            for win_idx in range(max(num_windows, 0)):
                start = win_idx * stride
                end = start + window_size
                query_start = end
                query_end = min(end + future_query_range, actual_len)
                
                if query_end - query_start < num_future_queries:
                    continue
                
                # Extract KV window (use first head for simplicity)
                # For multi-head, we'd need to either:
                # 1. Average across heads
                # 2. Process each head separately
                # 3. Use MultiHeadSidecar
                k_window = keys[batch_idx, 0, start:end, :]  # (window, d_head)
                v_window = values[batch_idx, 0, start:end, :]
                
                # Sample future query positions
                available_queries = query_end - query_start
                query_indices = torch.randperm(available_queries)[:num_future_queries]
                query_positions = query_indices + query_start
                
                # Extract queries from hidden states
                if hidden is not None:
                    queries = hidden[batch_idx, query_positions, :]
                else:
                    # Fallback: use zeros (will need Q projection during training)
                    queries = torch.zeros(num_future_queries, head_dim)
                
                windows.append({
                    "layer_idx": layer_idx,
                    "batch_idx": batch_idx,
                    "window_start": start,
                    "keys": k_window,
                    "values": v_window,
                    "queries": queries,
                    "head_dim": head_dim,
                    "num_heads": num_heads,
                })
    
    return windows


def save_trajectories(
    windows: List[Dict],
    output_dir: Path,
    checkpoint_idx: int,
    metadata: Dict,
):
    """Save collected windows to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to tensors
    data = {
        "windows": [],
        "metadata": metadata,
    }
    
    for w in windows:
        data["windows"].append({
            "layer_idx": w["layer_idx"],
            "window_start": w["window_start"],
            "keys": w["keys"],
            "values": w["values"],
            "queries": w["queries"],
        })
    
    filepath = output_dir / f"trajectories_{checkpoint_idx:05d}.pt"
    torch.save(data, filepath)
    
    print(f"Saved {len(windows)} windows to {filepath}")
    return filepath


def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TRAJECTORY COLLECTION")
    print("=" * 60)
    print(f"\nModel: {args.model_name}")
    print(f"Dataset: {args.dataset_name}/{args.dataset_config}")
    print(f"Output: {output_dir}")
    print(f"Window size: {args.window_size}")
    print(f"Num samples: {args.num_samples}")
    
    # Determine dtype
    if args.torch_dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[args.torch_dtype]
    
    print(f"Dtype: {torch_dtype}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    
    # Get model config
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads
    
    print(f"\nModel config:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Hidden: {hidden_size}")
    
    # Parse target layers
    if args.target_layers:
        target_layers = [int(x) for x in args.target_layers.split(",")]
    else:
        target_layers = None  # All layers
    
    print(f"  Target layers: {target_layers or 'all'}")
    
    # Load dataset
    print(f"\nLoading dataset {args.dataset_name}...")
    try:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading dataset with config: {e}")
        print("Trying without config...")
        dataset = load_dataset(
            args.dataset_name,
            split=args.dataset_split,
            trust_remote_code=True,
        )
    
    # Extract texts
    if args.text_column in dataset.column_names:
        texts = dataset[args.text_column]
    else:
        # Try common column names
        for col in ["text", "content", "document"]:
            if col in dataset.column_names:
                texts = dataset[col]
                break
        else:
            raise ValueError(f"Could not find text column. Available: {dataset.column_names}")
    
    # Limit samples
    if len(texts) > args.num_samples:
        indices = random.sample(range(len(texts)), args.num_samples)
        texts = [texts[i] for i in indices]
    
    print(f"Processing {len(texts)} samples")
    
    # Create dataset and dataloader
    text_dataset = TextDataset(
        texts,
        tokenizer,
        max_length=args.max_seq_length,
        min_length=args.min_seq_length,
    )
    
    print(f"After filtering: {len(text_dataset)} samples")
    
    dataloader = DataLoader(
        text_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True,
    )
    
    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "window_size": args.window_size,
        "stride": args.stride,
        "num_future_queries": args.num_future_queries,
        "future_query_range": args.future_query_range,
        "target_layers": target_layers,
        "max_seq_length": args.max_seq_length,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Collection loop
    all_windows = []
    checkpoint_idx = 0
    total_windows = 0
    
    device = next(model.parameters()).device
    
    print("\nCollecting trajectories...")
    pbar = tqdm(dataloader, desc="Processing batches")
    
    for batch in pbar:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Sample layers if specified (for efficiency)
        if args.sample_layers and target_layers is None:
            batch_layers = random.sample(range(num_layers), min(args.sample_layers, num_layers))
        else:
            batch_layers = target_layers
        
        # Extract KV states
        try:
            kv_states, hidden_states = extract_kv_from_model(
                model,
                input_ids,
                attention_mask,
                target_layers=batch_layers,
            )
        except Exception as e:
            print(f"\nError extracting KV: {e}")
            continue
        
        # Extract windows
        windows = extract_windows(
            kv_states,
            hidden_states,
            window_size=args.window_size,
            stride=args.stride,
            num_future_queries=args.num_future_queries,
            future_query_range=args.future_query_range,
            attention_mask=attention_mask.cpu(),
        )
        
        all_windows.extend(windows)
        total_windows += len(windows)
        
        pbar.set_postfix({"windows": total_windows})
        
        # Checkpoint
        if len(all_windows) >= args.save_every:
            save_trajectories(all_windows, output_dir, checkpoint_idx, metadata)
            all_windows = []
            checkpoint_idx += 1
        
        # Clear cache
        del kv_states, hidden_states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save remaining
    if all_windows:
        save_trajectories(all_windows, output_dir, checkpoint_idx, metadata)
    
    # Final stats
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total windows collected: {total_windows}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoints saved: {checkpoint_idx + 1}")
    
    # Create config for Sidecar
    sidecar_config = SidecarConfig(
        d_head=head_dim,
        n_heads=num_heads,
        window_size=args.window_size,
    )
    
    with open(output_dir / "sidecar_config.json", "w") as f:
        json.dump(sidecar_config.__dict__, f, indent=2, default=str)
    
    print(f"\nSidecar config saved to {output_dir / 'sidecar_config.json'}")
    print(f"\nTo train Sidecar:")
    print(f"  python scripts/train_sidecar.py \\")
    print(f"    --model_name {args.model_name} \\")
    print(f"    --trajectories_path {output_dir} \\")
    print(f"    --batch_size 32")


if __name__ == "__main__":
    main()

