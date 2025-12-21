#!/usr/bin/env python3
"""
Perplexity Evaluation Script
============================

Evaluate LST and baselines on perplexity metrics.

Usage:
    python scripts/benchmark/eval_perplexity.py \\
        --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --checkpoint ./checkpoints/lst/final.pt \\
        --methods dense,lst,mean,h2o,streaming,kvmerger,weightedkv,cam,tova

Methods:
    Eviction-based:
    - dense: Full KV cache (baseline)
    - streaming: StreamingLLM (Xiao et al.) - evicts middle
    - h2o: Heavy-Hitter Oracle (Zhang et al.) - keeps high-attention tokens
    - tova: Token Omission Via Attention (Oren et al.) - greedy eviction

    Merging-based:
    - mean: Simple mean pooling baseline
    - lst: Learned Super-Token compression (ours)
    - kvmerger: Gaussian kernel merging (Wang et al.)
    - weightedkv: Attention-weighted value merging (Yuan et al.)
    - cam: Cache Merging (Zhang et al., ICML 2024)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baselines import (
    H2O,
    TOVA,
    CaM,
    CaMConfig,
    H2OConfig,
    KVMerger,
    KVMergerConfig,
    StreamingLLM,
    StreamingLLMConfig,
    TOVAConfig,
    WeightedKV,
    WeightedKVConfig,
)
from src.LST.sidecar import SidecarPPL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_name: str, device: torch.device):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def load_sidecar(checkpoint_path: str, device: torch.device) -> SidecarPPL:
    """Load trained sidecar from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    sidecar = SidecarPPL(
        d_head=config["d_head"],
        window_size=config["window_size"],
        hidden_dim=config["hidden_dim"],
        num_encoder_layers=config["num_encoder_layers"],
    ).to(device)

    sidecar.load_state_dict(ckpt["model_state_dict"])
    sidecar.eval()

    return sidecar


def compress_cache_lst(
    cache: list[tuple[torch.Tensor, torch.Tensor]],
    sidecar: SidecarPPL,
    window_size: int,
    num_sink: int,
    num_recent: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Compress cache using LST sidecar."""
    compressed = []

    for k, v in cache:
        B, H, S, D = k.shape

        if S <= num_sink + num_recent + window_size:
            compressed.append((k, v))
            continue

        ks = k[:, :, :num_sink, :]
        vs = v[:, :, :num_sink, :]
        kr = k[:, :, -num_recent:, :]
        vr = v[:, :, -num_recent:, :]
        km = k[:, :, num_sink:-num_recent, :]
        vm = v[:, :, num_sink:-num_recent, :]

        M = km.shape[2]
        num_windows = M // window_size

        if num_windows == 0:
            compressed.append((k, v))
            continue

        trim_len = num_windows * window_size
        km_windows = km[:, :, :trim_len, :].view(B, H, num_windows, window_size, D)
        vm_windows = vm[:, :, :trim_len, :].view(B, H, num_windows, window_size, D)
        km_leftover = km[:, :, trim_len:, :]
        vm_leftover = vm[:, :, trim_len:, :]

        km_flat = km_windows.reshape(-1, window_size, D)
        vm_flat = vm_windows.reshape(-1, window_size, D)
        kv_concat = torch.cat([km_flat, vm_flat], dim=-1)

        with torch.no_grad():
            k_comp, v_comp = sidecar(kv_concat)

        k_comp = k_comp.view(B, H, num_windows, D)
        v_comp = v_comp.view(B, H, num_windows, D)

        k_new = torch.cat([ks, k_comp, km_leftover, kr], dim=2)
        v_new = torch.cat([vs, v_comp, vm_leftover, vr], dim=2)

        compressed.append((k_new, v_new))

    return compressed


def compress_cache_mean(
    cache: list[tuple[torch.Tensor, torch.Tensor]],
    window_size: int,
    num_sink: int,
    num_recent: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Compress cache using mean pooling baseline."""
    compressed = []

    for k, v in cache:
        B, H, S, D = k.shape

        if S <= num_sink + num_recent + window_size:
            compressed.append((k, v))
            continue

        ks = k[:, :, :num_sink, :]
        vs = v[:, :, :num_sink, :]
        kr = k[:, :, -num_recent:, :]
        vr = v[:, :, -num_recent:, :]
        km = k[:, :, num_sink:-num_recent, :]
        vm = v[:, :, num_sink:-num_recent, :]

        M = km.shape[2]
        num_windows = M // window_size

        if num_windows == 0:
            compressed.append((k, v))
            continue

        trim_len = num_windows * window_size
        km_windows = km[:, :, :trim_len, :].view(B, H, num_windows, window_size, D)
        vm_windows = vm[:, :, :trim_len, :].view(B, H, num_windows, window_size, D)
        km_leftover = km[:, :, trim_len:, :]
        vm_leftover = vm[:, :, trim_len:, :]

        # Simple mean pooling
        k_comp = km_windows.mean(dim=3)
        v_comp = vm_windows.mean(dim=3)

        k_new = torch.cat([ks, k_comp, km_leftover, kr], dim=2)
        v_new = torch.cat([vs, v_comp, vm_leftover, vr], dim=2)

        compressed.append((k_new, v_new))

    return compressed


def compress_cache_with_baseline(
    cache: list[tuple[torch.Tensor, torch.Tensor]],
    method_name: str,
    budget: int,
    num_sink: int,
    num_recent: int,
    **kwargs,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Generic cache compression using baseline methods.

    Supports: h2o, streaming, tova, kvmerger, weightedkv, cam
    """
    compressed = []

    # Create method instance based on name
    if method_name == "h2o":
        config = H2OConfig(num_sink=num_sink, num_recent=num_recent, budget=budget)
        method = H2O(config)
    elif method_name == "streaming":
        config = StreamingLLMConfig(num_sink=num_sink, num_recent=num_recent)
        method = StreamingLLM(config)
    elif method_name == "tova":
        config = TOVAConfig(num_sink=num_sink, num_recent=num_recent, budget=budget)
        method = TOVA(config)
    elif method_name == "kvmerger":
        config = KVMergerConfig(num_sink=num_sink, num_recent=num_recent)
        method = KVMerger(config)
    elif method_name == "weightedkv":
        config = WeightedKVConfig(num_sink=num_sink, num_recent=num_recent, budget=budget)
        method = WeightedKV(config)
    elif method_name == "cam":
        config = CaMConfig(num_sink=num_sink, num_recent=num_recent, budget=budget)
        method = CaM(config)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    for k, v in cache:
        B, H, S, D = k.shape

        # Reshape to (B*H, S, D) for baseline methods
        k_flat = k.transpose(1, 2).reshape(B, S, H * D)
        v_flat = v.transpose(1, 2).reshape(B, S, H * D)

        # Apply compression
        k_comp, v_comp = method.compress(k_flat, v_flat)

        # Reshape back to (B, H, S', D)
        S_new = k_comp.shape[1]
        k_comp = k_comp.view(B, S_new, H, D).transpose(1, 2)
        v_comp = v_comp.view(B, S_new, H, D).transpose(1, 2)

        compressed.append((k_comp, v_comp))

    return compressed


def evaluate_ppl(
    model,
    tokenizer,
    samples: list[str],
    method: str,
    sidecar: SidecarPPL | None = None,
    window_size: int = 8,
    num_sink: int = 4,
    num_recent: int = 8,
    prefix_ratio: float = 0.5,
) -> dict[str, float]:
    """Evaluate perplexity for a given compression method."""
    device = next(model.parameters()).device
    losses = []

    for text in tqdm(samples, desc=f"Evaluating {method}"):
        tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)

        if input_ids.shape[1] < 100:
            continue

        # Split into prefix and suffix
        prefix_len = int(input_ids.shape[1] * prefix_ratio)
        prefix = input_ids[:, :prefix_len]
        suffix = input_ids[:, prefix_len:]

        # Get cache from prefix
        with torch.no_grad():
            outputs = model(prefix, use_cache=True)
            cache = list(outputs.past_key_values)

        # Compute budget for eviction methods
        budget = num_sink + num_recent + (prefix_len - num_sink - num_recent) // window_size

        # Compress cache based on method
        if method == "dense":
            compressed = cache
        elif method == "lst":
            compressed = compress_cache_lst(cache, sidecar, window_size, num_sink, num_recent)
        elif method == "mean":
            compressed = compress_cache_mean(cache, window_size, num_sink, num_recent)
        elif method in ["h2o", "streaming", "tova", "kvmerger", "weightedkv", "cam"]:
            compressed = compress_cache_with_baseline(cache, method, budget, num_sink, num_recent)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute loss on suffix
        with torch.no_grad():
            labels = suffix.clone()
            labels[:, 0] = -100
            outputs = model(suffix, past_key_values=tuple(compressed), labels=labels)
            losses.append(outputs.loss.item())

    mean_loss = np.mean(losses)
    ppl = np.exp(mean_loss)

    return {"loss": mean_loss, "ppl": ppl, "n_samples": len(losses)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate LST and baselines on PPL")

    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="LST checkpoint path")
    parser.add_argument(
        "--methods",
        type=str,
        default="dense,lst,mean,h2o,streaming,tova,kvmerger,weightedkv,cam",
        help="Comma-separated list of methods",
    )
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--window_size", type=int, default=8, help="Compression window size")
    parser.add_argument("--num_sink", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--num_recent", type=int, default=8, help="Number of recent tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model(args.model_name, device)

    # Load sidecar if needed
    sidecar = None
    methods = args.methods.split(",")
    if "lst" in methods:
        if args.checkpoint is None:
            logger.warning("No checkpoint provided for LST, skipping")
            methods.remove("lst")
        else:
            sidecar = load_sidecar(args.checkpoint, device)

    # Load evaluation data
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)
    dataset = dataset.shuffle(seed=args.seed)

    samples = []
    for item in dataset:
        if len(item["text"].strip()) > 200:
            samples.append(item["text"])
        if len(samples) >= args.num_samples:
            break

    logger.info(f"Loaded {len(samples)} evaluation samples")

    # Evaluate each method
    results = {}
    for method in methods:
        logger.info(f"Evaluating method: {method}")
        try:
            result = evaluate_ppl(
                model,
                tokenizer,
                samples,
                method,
                sidecar=sidecar,
                window_size=args.window_size,
                num_sink=args.num_sink,
                num_recent=args.num_recent,
            )
            results[method] = result
            logger.info(f"  {method}: PPL={result['ppl']:.2f}")
        except Exception as e:
            logger.error(f"  {method}: FAILED - {e}")
            results[method] = {"loss": float("nan"), "ppl": float("nan"), "n_samples": 0}

    # Print summary
    print("\n" + "=" * 60)
    print("PERPLEXITY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Window size: {args.window_size} (8:1 compression)")
    print(f"Sink tokens: {args.num_sink}")
    print(f"Recent tokens: {args.num_recent}")
    print("-" * 60)
    print(f"{'Method':<15} {'Type':<12} {'PPL':>10} {'Loss':>10}")
    print("-" * 60)

    method_types = {
        "dense": "baseline",
        "streaming": "eviction",
        "h2o": "eviction",
        "tova": "eviction",
        "mean": "merging",
        "lst": "merging",
        "kvmerger": "merging",
        "weightedkv": "merging",
        "cam": "merging",
    }

    for method, result in results.items():
        mtype = method_types.get(method, "unknown")
        ppl_str = f"{result['ppl']:.2f}" if not np.isnan(result["ppl"]) else "FAILED"
        loss_str = f"{result['loss']:.4f}" if not np.isnan(result["loss"]) else "FAILED"
        print(f"{method:<15} {mtype:<12} {ppl_str:>10} {loss_str:>10}")

    print("=" * 60)


if __name__ == "__main__":
    main()
