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
import json
import logging
import re
import sys
from datetime import datetime
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

    # Match model dtype (bfloat16 on CUDA)
    if torch.cuda.is_available():
        sidecar = sidecar.to(torch.bfloat16)

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

    Following the standard NIAH methodology (gkamradt/LLMTest_NeedleInAHaystack):
    - Depth 0.0 = needle at the very beginning
    - Depth 1.0 = needle at the very end
    - Needle is NEVER trimmed (we build haystack around it)

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
    needle_tokens = len(tokenizer.encode(needle))

    # Calculate how many tokens we need for haystack (excluding needle)
    haystack_budget = target_length - needle_tokens - 10  # small buffer

    # Build pre-needle and post-needle haystack based on depth
    # depth=0.0 means needle at start (0% pre, 100% post)
    # depth=1.0 means needle at end (100% pre, 0% post)
    pre_budget = int(haystack_budget * needle_depth)
    post_budget = haystack_budget - pre_budget

    # Build pre-needle text
    pre_sentences = []
    pre_length = 0
    while pre_length < pre_budget:
        sentence = rng.choice(HAYSTACK_SENTENCES)
        sentence_len = len(tokenizer.encode(sentence))
        if pre_length + sentence_len > pre_budget + 20:  # small overflow ok
            break
        pre_sentences.append(sentence)
        pre_length += sentence_len

    # Build post-needle text
    post_sentences = []
    post_length = 0
    while post_length < post_budget:
        sentence = rng.choice(HAYSTACK_SENTENCES)
        sentence_len = len(tokenizer.encode(sentence))
        if post_length + sentence_len > post_budget + 20:  # small overflow ok
            break
        post_sentences.append(sentence)
        post_length += sentence_len

    # Combine: pre + needle + post
    pre_text = " ".join(pre_sentences)
    post_text = " ".join(post_sentences)

    if pre_text and post_text:
        full_text = f"{pre_text} {needle} {post_text}"
    elif pre_text:
        full_text = f"{pre_text} {needle}"
    elif post_text:
        full_text = f"{needle} {post_text}"
    else:
        full_text = needle

    # Final trim if needed (trim from END, never touching needle)
    tokens = tokenizer.encode(full_text)
    if len(tokens) > target_length:
        # Find needle position in tokens to ensure we don't cut it
        needle_start = full_text.find(needle)
        needle_end = needle_start + len(needle)

        # Only trim post-needle portion
        pre_needle_text = full_text[:needle_end]
        post_needle_text = full_text[needle_end:]

        # Calculate how much to trim from post
        pre_tokens = len(tokenizer.encode(pre_needle_text))
        available_post = target_length - pre_tokens
        if available_post > 0:
            post_tokens = tokenizer.encode(post_needle_text)[:available_post]
            post_needle_text = tokenizer.decode(post_tokens, skip_special_tokens=True)
            full_text = pre_needle_text + post_needle_text
        else:
            # Edge case: even needle alone exceeds target
            full_text = pre_needle_text

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
    context_len: int,
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
        context_len: Length of context tokens (for cache_position)
        max_new_tokens: Max tokens to generate

    Returns:
        Tuple of (is_correct, generated_answer)
    """
    from transformers.cache_utils import DynamicCache

    device = next(model.parameters()).device

    # Tokenize question
    question_tokens = tokenizer(
        f"\n\nQuestion: {question}\nAnswer:",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    # Convert cache list to DynamicCache for newer transformers
    past_kv = DynamicCache()
    for k, v in cache:
        past_kv.update(k, v, layer_idx=len(past_kv))

    # Get cache length from the actual compressed cache
    cache_len = cache[0][0].shape[2] if cache else 0

    # Manual autoregressive generation (works better with external cache)
    all_tokens = question_tokens.clone()

    with torch.no_grad():
        # Process all question tokens in ONE forward pass (not token-by-token)
        q_len = question_tokens.shape[1]
        position_ids = torch.arange(cache_len, cache_len + q_len, device=device).unsqueeze(0)
        out = model(
            question_tokens,
            past_key_values=past_kv,
            position_ids=position_ids,
            use_cache=True,
        )
        past_kv = out.past_key_values

        # Then generate new tokens
        for _ in range(max_new_tokens):
            pos = cache_len + all_tokens.shape[1]
            position_ids = torch.tensor([[pos]], device=device)
            out = model(
                all_tokens[:, -1:],
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            all_tokens = torch.cat([all_tokens, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode response
    response = tokenizer.decode(
        all_tokens[0, question_tokens.shape[1] :],
        skip_special_tokens=True,
    ).strip()

    # Check if answer is correct using multiple matching strategies
    # 1. Exact substring match (standard)
    # 2. Extract first number from response and compare (robust to formatting)
    is_correct = False

    # Method 1: Direct substring match
    if expected_answer in response:
        is_correct = True
    else:
        # Method 2: Extract numbers from response and check
        numbers_in_response = re.findall(r'\b\d{4}\b', response)  # 4-digit numbers
        if expected_answer in numbers_in_response:
            is_correct = True

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
                    past_kv = outputs.past_key_values

                    # Convert DynamicCache to list of tuples if needed
                    if hasattr(past_kv, "key_cache"):
                        cache = [
                            (past_kv.key_cache[i], past_kv.value_cache[i])
                            for i in range(len(past_kv.key_cache))
                        ]
                    else:
                        cache = list(past_kv)

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
                    model, tokenizer, context, QUESTION_TEMPLATE, expected, compressed,
                    context_len=prefix_len,
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
    parser = argparse.ArgumentParser(
        description="Needle-in-a-Haystack Evaluation (following gkamradt methodology)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 samples)
  python eval_niah.py --methods dense,mean --num_samples 5

  # ICML-quality evaluation (50 samples per config)
  python eval_niah.py --methods dense,lst,h2o,streaming --num_samples 50 \\
      --context_lengths 1024,2048,4096 --output results/niah_results.json

  # Single compression ratio sweep
  python eval_niah.py --window_size 4 --num_samples 20 --output results/niah_4x.json
        """,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="LST checkpoint path")
    parser.add_argument(
        "--methods",
        type=str,
        default="dense,lst,mean,h2o,streaming,tova,kvmerger,weightedkv,cam",
        help="Comma-separated methods: dense,lst,mean,h2o,streaming,tova,kvmerger,weightedkv,cam",
    )
    parser.add_argument(
        "--context_lengths",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated context lengths (standard: 1024,2048,4096,8192)",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated needle depths (0.0=start, 1.0=end)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Samples per (length, depth) config. ICML standard: 50-100",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Compression window size (e.g., 2=2:1, 4=4:1, 8=8:1)",
    )
    parser.add_argument("--num_sink", type=int, default=4, help="Sink tokens to preserve")
    parser.add_argument("--num_recent", type=int, default=8, help="Recent tokens to preserve")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (for reproducibility)",
    )

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

    # Save results to JSON for reproducibility
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tuple keys to strings for JSON serialization
        json_results = {}
        for method, result in all_results.items():
            json_results[method] = {
                "overall_accuracy": result.get("overall_accuracy", 0),
                "total_correct": result.get("total_correct", 0),
                "total_samples": result.get("total_samples", 0),
            }
            if "per_config" in result:
                json_results[method]["per_config"] = {
                    f"{length}_{depth}": acc
                    for (length, depth), acc in result["per_config"].items()
                }
            if "error" in result:
                json_results[method]["error"] = result["error"]

        output_data = {
            "metadata": {
                "model": args.model_name,
                "window_size": args.window_size,
                "compression_ratio": f"{args.window_size}:1",
                "context_lengths": context_lengths,
                "depths": depths,
                "num_samples": args.num_samples,
                "num_sink": args.num_sink,
                "num_recent": args.num_recent,
                "seed": args.seed,
                "timestamp": datetime.now().isoformat(),
            },
            "results": json_results,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
