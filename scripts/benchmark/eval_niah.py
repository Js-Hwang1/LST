#!/usr/bin/env python3
"""
Needle-in-a-Haystack (NIAH) Evaluation Script
==============================================

Evaluate LST and baselines on the Needle-in-a-Haystack retrieval task.
This tests whether compressed KV caches can accurately retrieve information
placed at various depths within long contexts.

Usage:
    python scripts/benchmark/eval_niah.py \\
        --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --checkpoint ./checkpoints/lst/final.pt \\
        --methods dense,lst,mean,h2o,streaming

The benchmark:
    1. Creates a "haystack" of random text with a "needle" (key fact) inserted
    2. Places needle at different depths (0%, 25%, 50%, 75%, 100%)
    3. Tests at different context lengths (512, 1024, 2048, etc.)
    4. Measures retrieval accuracy for each (depth, length) combination
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
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

# Needle template
NEEDLE_TEMPLATE = "The special magic number is {number}."
QUESTION_TEMPLATE = "What is the special magic number mentioned in the text above?"

# Haystack filler (Paul Graham essays style)
HAYSTACK_SENTENCES = [
    "The most important thing is to build something users love.",
    "Startups are all about growth.",
    "Great hackers tend to clump together.",
    "The best ideas often seem absurd at first.",
    "You make what you measure.",
    "Do things that don't scale.",
    "The most successful startups have deep technical insight.",
    "Speed is the most important factor in startup success.",
    "The best way to have startup ideas is to be at the leading edge.",
    "Wealth is what you want, not money.",
    "The way to get startup ideas is to look for problems.",
    "Write code and talk to users.",
    "Programming is a way of thinking, not just a skill.",
    "The best test of a startup idea is whether it has good word of mouth.",
    "If you have a good idea, it will have lots of competition.",
]


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


def generate_haystack(
    tokenizer,
    target_length: int,
    needle_depth: float,
    magic_number: int,
    seed: int = 42,
) -> tuple[str, str]:
    """
    Generate a haystack with a needle inserted at specified depth.

    Args:
        tokenizer: Tokenizer for length estimation
        target_length: Target length in tokens
        needle_depth: Where to insert needle (0.0=start, 1.0=end)
        magic_number: The number to hide in the needle
        seed: Random seed for reproducibility

    Returns:
        Tuple of (full_text, expected_answer)
    """
    rng = np.random.RandomState(seed)

    needle = NEEDLE_TEMPLATE.format(number=magic_number)

    # Build haystack
    sentences = []
    current_length = 0

    while current_length < target_length * 1.5:  # Overshoot then trim
        sentence = rng.choice(HAYSTACK_SENTENCES)
        sentences.append(sentence)
        current_length += len(tokenizer.encode(sentence))

    # Find insertion point
    total_sentences = len(sentences)
    insert_idx = int(total_sentences * needle_depth)
    insert_idx = max(0, min(insert_idx, total_sentences - 1))

    # Insert needle
    sentences.insert(insert_idx, needle)

    # Join and trim to target length
    full_text = " ".join(sentences)
    tokens = tokenizer.encode(full_text)

    if len(tokens) > target_length:
        # Trim from end, keeping needle
        tokens = tokens[:target_length]
        full_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return full_text, str(magic_number)


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
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generic cache compression using baseline methods."""
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

    compressed = []

    for k, v in cache:
        B, H, S, D = k.shape

        k_flat = k.transpose(1, 2).reshape(B, S, H * D)
        v_flat = v.transpose(1, 2).reshape(B, S, H * D)

        k_comp, v_comp = method.compress(k_flat, v_flat)

        S_new = k_comp.shape[1]
        k_comp = k_comp.view(B, S_new, H, D).transpose(1, 2)
        v_comp = v_comp.view(B, S_new, H, D).transpose(1, 2)

        compressed.append((k_comp, v_comp))

    return compressed


def evaluate_retrieval(
    model,
    tokenizer,
    context: str,
    question: str,
    expected_answer: str,
    cache: list[tuple[torch.Tensor, torch.Tensor]],
    max_new_tokens: int = 20,
) -> tuple[bool, str]:
    """
    Evaluate if model can retrieve the answer using compressed cache.

    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Context with needle
        question: Question to ask
        expected_answer: Expected answer
        cache: Compressed KV cache
        max_new_tokens: Max tokens to generate

    Returns:
        Tuple of (is_correct, generated_answer)
    """
    device = next(model.parameters()).device

    # Tokenize question
    question_tokens = tokenizer(
        f"\n\nQuestion: {question}\nAnswer:",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    # Generate with compressed cache
    with torch.no_grad():
        generated = model.generate(
            question_tokens,
            past_key_values=tuple(cache),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode response
    response = tokenizer.decode(
        generated[0, question_tokens.shape[1] :],
        skip_special_tokens=True,
    ).strip()

    # Check if answer is present
    is_correct = expected_answer in response

    return is_correct, response


def evaluate_niah(
    model,
    tokenizer,
    method: str,
    sidecar: SidecarPPL | None = None,
    context_lengths: list[int] = None,
    depths: list[float] = None,
    window_size: int = 8,
    num_sink: int = 4,
    num_recent: int = 8,
    num_samples: int = 5,
    seed: int = 42,
) -> dict:
    """
    Run full NIAH evaluation.

    Args:
        model: Language model
        tokenizer: Tokenizer
        method: Compression method
        sidecar: Trained sidecar (for LST)
        context_lengths: List of context lengths to test
        depths: List of needle depths to test
        window_size: Window size for compression
        num_sink: Sink tokens
        num_recent: Recent tokens
        num_samples: Samples per (length, depth) combination
        seed: Random seed

    Returns:
        Results dictionary
    """
    if context_lengths is None:
        context_lengths = [256, 512, 1024]
    if depths is None:
        depths = [0.0, 0.25, 0.5, 0.75, 1.0]

    device = next(model.parameters()).device
    results = {}
    rng = np.random.RandomState(seed)

    total_correct = 0
    total_samples = 0

    for length in tqdm(context_lengths, desc=f"Evaluating {method}"):
        for depth in depths:
            correct = 0

            for sample_idx in range(num_samples):
                magic_number = rng.randint(1000, 9999)
                sample_seed = seed + sample_idx + int(length * 100 + depth * 1000)

                # Generate haystack
                context, expected = generate_haystack(
                    tokenizer, length, depth, magic_number, sample_seed
                )

                # Tokenize context
                context_tokens = tokenizer(
                    context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=length,
                ).input_ids.to(device)

                # Get cache from context
                with torch.no_grad():
                    outputs = model(context_tokens, use_cache=True)
                    cache = list(outputs.past_key_values)

                # Compute budget
                prefix_len = context_tokens.shape[1]
                budget = num_sink + num_recent + (prefix_len - num_sink - num_recent) // window_size

                # Compress cache
                if method == "dense":
                    compressed = cache
                elif method == "lst":
                    compressed = compress_cache_lst(
                        cache, sidecar, window_size, num_sink, num_recent
                    )
                elif method == "mean":
                    compressed = compress_cache_mean(cache, window_size, num_sink, num_recent)
                else:
                    compressed = compress_cache_with_baseline(
                        cache, method, budget, num_sink, num_recent
                    )

                # Evaluate retrieval
                is_correct, response = evaluate_retrieval(
                    model, tokenizer, context, QUESTION_TEMPLATE, expected, compressed
                )

                if is_correct:
                    correct += 1

            accuracy = correct / num_samples
            results[(length, depth)] = accuracy
            total_correct += correct
            total_samples += num_samples

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    return {
        "per_config": results,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Needle-in-a-Haystack")

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
    parser.add_argument(
        "--context_lengths",
        type=str,
        default="256,512,1024",
        help="Comma-separated context lengths",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated needle depths",
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Samples per config")
    parser.add_argument("--window_size", type=int, default=8, help="Compression window size")
    parser.add_argument("--num_sink", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--num_recent", type=int, default=8, help="Number of recent tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Parse lists
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

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

    # Evaluate each method
    all_results = {}
    for method in methods:
        logger.info(f"Evaluating method: {method}")
        try:
            result = evaluate_niah(
                model,
                tokenizer,
                method,
                sidecar=sidecar,
                context_lengths=context_lengths,
                depths=depths,
                window_size=args.window_size,
                num_sink=args.num_sink,
                num_recent=args.num_recent,
                num_samples=args.num_samples,
                seed=args.seed,
            )
            all_results[method] = result
            logger.info(f"  {method}: Accuracy={result['overall_accuracy']:.2%}")
        except Exception as e:
            logger.error(f"  {method}: FAILED - {e}")
            all_results[method] = {"overall_accuracy": 0.0, "error": str(e)}

    # Print summary
    print("\n" + "=" * 70)
    print("NEEDLE-IN-A-HAYSTACK EVALUATION RESULTS")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Context lengths: {context_lengths}")
    print(f"Depths: {depths}")
    print(f"Samples per config: {args.num_samples}")
    print(f"Window size: {args.window_size} (8:1 compression)")
    print("-" * 70)

    # Overall accuracy table
    print(f"\n{'Method':<15} {'Accuracy':>10}")
    print("-" * 25)
    for method, result in all_results.items():
        acc = result.get("overall_accuracy", 0)
        print(f"{method:<15} {acc:>10.1%}")

    # Detailed heatmap-style results
    print("\n" + "-" * 70)
    print("Accuracy by (Context Length, Depth)")
    print("-" * 70)

    for method, result in all_results.items():
        if "per_config" not in result:
            continue
        print(f"\n{method}:")
        # Header
        header = "Length\\Depth"
        for d in depths:
            header += f"  {d:.2f}"
        print(header)

        for length in context_lengths:
            row = f"{length:>6}"
            for d in depths:
                acc = result["per_config"].get((length, d), 0)
                row += f"  {acc:.2f}"
            print(row)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
