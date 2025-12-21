#!/usr/bin/env python3
"""
LongBench Evaluation Script
============================

Evaluate LST and baselines on LongBench, a comprehensive benchmark
for long-context language understanding.

Usage:
    python scripts/benchmark/eval_longbench.py \\
        --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --checkpoint ./checkpoints/lst/final.pt \\
        --methods dense,lst,mean,h2o,streaming \\
        --tasks narrativeqa,qasper

LongBench Tasks (grouped by category):
    Single-Document QA:
        - narrativeqa: Story comprehension
        - qasper: Scientific paper QA
        - multifieldqa_en: Multi-domain QA

    Multi-Document QA:
        - hotpotqa: Multi-hop reasoning
        - 2wikimqa: 2-Wikipedia multi-hop
        - musique: Multi-step reasoning

    Summarization:
        - gov_report: Government report summarization
        - qmsum: Meeting summarization
        - multi_news: Multi-document news summarization

    Few-shot Learning:
        - trec: Question classification
        - triviaqa: Trivia questions
        - samsum: Dialogue summarization

    Synthetic:
        - passage_count: Count passages containing keyword
        - passage_retrieval_en: Retrieve relevant passage

    Code:
        - lcc: Long code completion
        - repobench-p: Repository-level code completion

Reference: Bai et al. "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.LST.sidecar import SidecarPPL
from src.baselines import (
    H2O, H2OConfig,
    StreamingLLM, StreamingLLMConfig,
    TOVA, TOVAConfig,
    KVMerger, KVMergerConfig,
    WeightedKV, WeightedKVConfig,
    CaM, CaMConfig,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Task configurations
TASK_CONFIGS = {
    # Single-doc QA
    "narrativeqa": {"type": "qa", "metric": "f1", "max_gen": 128},
    "qasper": {"type": "qa", "metric": "f1", "max_gen": 128},
    "multifieldqa_en": {"type": "qa", "metric": "f1", "max_gen": 64},
    # Multi-doc QA
    "hotpotqa": {"type": "qa", "metric": "f1", "max_gen": 32},
    "2wikimqa": {"type": "qa", "metric": "f1", "max_gen": 32},
    "musique": {"type": "qa", "metric": "f1", "max_gen": 32},
    # Summarization
    "gov_report": {"type": "summarization", "metric": "rouge", "max_gen": 512},
    "qmsum": {"type": "summarization", "metric": "rouge", "max_gen": 256},
    "multi_news": {"type": "summarization", "metric": "rouge", "max_gen": 256},
    # Few-shot
    "trec": {"type": "classification", "metric": "accuracy", "max_gen": 16},
    "triviaqa": {"type": "qa", "metric": "f1", "max_gen": 32},
    "samsum": {"type": "summarization", "metric": "rouge", "max_gen": 128},
    # Synthetic
    "passage_count": {"type": "counting", "metric": "accuracy", "max_gen": 8},
    "passage_retrieval_en": {"type": "retrieval", "metric": "accuracy", "max_gen": 32},
    # Code
    "lcc": {"type": "code", "metric": "edit_sim", "max_gen": 64},
    "repobench-p": {"type": "code", "metric": "edit_sim", "max_gen": 64},
}

DEFAULT_TASKS = ["narrativeqa", "hotpotqa", "gov_report", "trec", "passage_count"]


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


def load_longbench_task(task_name: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    Load a LongBench task from HuggingFace datasets.

    Args:
        task_name: Name of the task
        num_samples: Maximum number of samples to load

    Returns:
        List of samples with 'input', 'context', 'answers' fields
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "THUDM/LongBench",
            task_name,
            split="test",
            trust_remote_code=True,
        )

        samples = []
        for item in dataset:
            sample = {
                "input": item.get("input", ""),
                "context": item.get("context", ""),
                "answers": item.get("answers", []),
            }
            # Some tasks have 'question' instead of 'input'
            if not sample["input"] and "question" in item:
                sample["input"] = item["question"]

            samples.append(sample)

            if num_samples and len(samples) >= num_samples:
                break

        return samples

    except Exception as e:
        logger.error(f"Failed to load task {task_name}: {e}")
        return []


def compress_cache_lst(
    cache: List[Tuple[torch.Tensor, torch.Tensor]],
    sidecar: SidecarPPL,
    window_size: int,
    num_sink: int,
    num_recent: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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
    cache: List[Tuple[torch.Tensor, torch.Tensor]],
    window_size: int,
    num_sink: int,
    num_recent: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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
    cache: List[Tuple[torch.Tensor, torch.Tensor]],
    method_name: str,
    budget: int,
    num_sink: int,
    num_recent: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    import re
    import string

    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()

    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue

        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def compute_rouge_l(prediction: str, ground_truths: List[str]) -> float:
    """Compute ROUGE-L score."""
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    pred_tokens = prediction.lower().split()
    if not pred_tokens:
        return 0.0

    best_score = 0.0
    for gt in ground_truths:
        gt_tokens = gt.lower().split()
        if not gt_tokens:
            continue

        lcs = lcs_length(pred_tokens, gt_tokens)
        precision = lcs / len(pred_tokens) if pred_tokens else 0
        recall = lcs / len(gt_tokens) if gt_tokens else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_score = max(best_score, f1)

    return best_score


def compute_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """Compute exact match accuracy."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == pred_norm:
            return 1.0
    return 0.0


def compute_edit_similarity(prediction: str, ground_truths: List[str]) -> float:
    """Compute edit similarity for code completion."""
    def edit_distance(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    best_sim = 0.0
    for gt in ground_truths:
        max_len = max(len(prediction), len(gt))
        if max_len == 0:
            continue
        dist = edit_distance(prediction, gt)
        sim = 1 - dist / max_len
        best_sim = max(best_sim, sim)

    return best_sim


def evaluate_sample(
    model,
    tokenizer,
    sample: Dict,
    method: str,
    sidecar: Optional[SidecarPPL],
    task_config: Dict,
    window_size: int,
    num_sink: int,
    num_recent: int,
    max_context_length: int,
) -> Tuple[float, str]:
    """
    Evaluate a single sample.

    Returns:
        Tuple of (score, generated_text)
    """
    device = next(model.parameters()).device

    # Build prompt
    context = sample["context"]
    question = sample["input"]

    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    # Tokenize and truncate
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_length,
    )
    input_ids = tokens.input_ids.to(device)

    # Get cache from context (all but last few tokens)
    context_len = max(1, input_ids.shape[1] - 20)
    context_ids = input_ids[:, :context_len]
    question_ids = input_ids[:, context_len:]

    with torch.no_grad():
        outputs = model(context_ids, use_cache=True)
        cache = list(outputs.past_key_values)

    # Compute budget
    budget = num_sink + num_recent + (context_len - num_sink - num_recent) // window_size

    # Compress cache
    if method == "dense":
        compressed = cache
    elif method == "lst":
        compressed = compress_cache_lst(cache, sidecar, window_size, num_sink, num_recent)
    elif method == "mean":
        compressed = compress_cache_mean(cache, window_size, num_sink, num_recent)
    else:
        compressed = compress_cache_with_baseline(
            cache, method, budget, num_sink, num_recent
        )

    # Generate with compressed cache
    max_new_tokens = task_config.get("max_gen", 64)

    with torch.no_grad():
        generated = model.generate(
            question_ids,
            past_key_values=tuple(compressed),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode response
    response = tokenizer.decode(
        generated[0, question_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Compute score based on metric
    answers = sample.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]

    metric = task_config.get("metric", "f1")
    if metric == "f1":
        score = compute_f1(response, answers)
    elif metric == "rouge":
        score = compute_rouge_l(response, answers)
    elif metric == "accuracy":
        score = compute_accuracy(response, answers)
    elif metric == "edit_sim":
        score = compute_edit_similarity(response, answers)
    else:
        score = 0.0

    return score, response


def evaluate_task(
    model,
    tokenizer,
    task_name: str,
    method: str,
    sidecar: Optional[SidecarPPL],
    window_size: int,
    num_sink: int,
    num_recent: int,
    num_samples: int,
    max_context_length: int,
) -> Dict:
    """Evaluate a single LongBench task."""
    task_config = TASK_CONFIGS.get(task_name, {"type": "qa", "metric": "f1", "max_gen": 64})

    # Load data
    samples = load_longbench_task(task_name, num_samples)
    if not samples:
        return {"score": 0.0, "n_samples": 0, "error": "Failed to load data"}

    scores = []
    for sample in tqdm(samples, desc=f"{task_name}", leave=False):
        try:
            score, _ = evaluate_sample(
                model, tokenizer, sample, method, sidecar, task_config,
                window_size, num_sink, num_recent, max_context_length
            )
            scores.append(score)
        except Exception as e:
            logger.warning(f"Sample failed: {e}")
            scores.append(0.0)

    mean_score = np.mean(scores) if scores else 0.0

    return {
        "score": mean_score,
        "n_samples": len(scores),
        "metric": task_config["metric"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on LongBench")

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
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated list of tasks",
    )
    parser.add_argument("--num_samples", type=int, default=50, help="Samples per task")
    parser.add_argument("--max_context_length", type=int, default=2048, help="Max context length")
    parser.add_argument("--window_size", type=int, default=8, help="Compression window size")
    parser.add_argument("--num_sink", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--num_recent", type=int, default=8, help="Number of recent tokens")
    parser.add_argument("--output_file", type=str, default=None, help="Save results to JSON")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Parse lists
    tasks = args.tasks.split(",")
    methods = args.methods.split(",")

    # Validate tasks
    for task in tasks:
        if task not in TASK_CONFIGS:
            logger.warning(f"Unknown task: {task}, skipping")
            tasks.remove(task)

    # Load model
    model, tokenizer = load_model(args.model_name, device)

    # Load sidecar if needed
    sidecar = None
    if "lst" in methods:
        if args.checkpoint is None:
            logger.warning("No checkpoint provided for LST, skipping")
            methods.remove("lst")
        else:
            sidecar = load_sidecar(args.checkpoint, device)

    # Evaluate
    all_results = {}

    for method in methods:
        logger.info(f"Evaluating method: {method}")
        method_results = {}

        for task in tasks:
            logger.info(f"  Task: {task}")
            try:
                result = evaluate_task(
                    model, tokenizer, task, method, sidecar,
                    args.window_size, args.num_sink, args.num_recent,
                    args.num_samples, args.max_context_length
                )
                method_results[task] = result
                logger.info(f"    Score: {result['score']:.3f} ({result['metric']})")
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                method_results[task] = {"score": 0.0, "error": str(e)}

        # Compute average
        scores = [r["score"] for r in method_results.values() if "score" in r]
        method_results["average"] = np.mean(scores) if scores else 0.0

        all_results[method] = method_results

    # Print summary
    print("\n" + "=" * 80)
    print("LONGBENCH EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Tasks: {tasks}")
    print(f"Samples per task: {args.num_samples}")
    print(f"Window size: {args.window_size} (8:1 compression)")
    print("-" * 80)

    # Results table
    header = f"{'Method':<15}"
    for task in tasks:
        header += f" {task[:12]:>12}"
    header += f" {'Average':>10}"
    print(header)
    print("-" * 80)

    for method, results in all_results.items():
        row = f"{method:<15}"
        for task in tasks:
            score = results.get(task, {}).get("score", 0)
            row += f" {score:>12.3f}"
        avg = results.get("average", 0)
        row += f" {avg:>10.3f}"
        print(row)

    print("=" * 80)

    # Save results
    if args.output_file:
        output = {
            "model": args.model_name,
            "tasks": tasks,
            "methods": methods,
            "results": all_results,
        }
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
