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
    PyramidKV,
    PyramidKVConfig,
    SnapKV,
    SnapKVConfig,
    StreamingLLM,
    StreamingLLMConfig,
    TOVAConfig,
    WeightedKV,
    WeightedKVConfig,
)
from src.LST.sidecar import SidecarPPL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def load_sidecar(
    checkpoint_path: str, device: torch.device, model_config=None
) -> SidecarPPL:
    """Load trained sidecar from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Handle d_head being None in older checkpoints
    d_head = config.get("d_head")
    if d_head is None and model_config is not None:
        d_head = getattr(model_config, "head_dim", None)
        if d_head is None:
            d_head = model_config.hidden_size // model_config.num_attention_heads
        logger.info(f"Computed d_head from model config: {d_head}")

    if d_head is None:
        raise ValueError("d_head is None and no model_config provided to compute it")

    sidecar = SidecarPPL(
        d_head=d_head,
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


def load_longbench_task(task_name: str, num_samples: int | None = None) -> list[dict]:
    """
    Load a LongBench task from HuggingFace Hub.

    Downloads data.zip and extracts JSONL files.

    Args:
        task_name: Name of the task
        num_samples: Maximum number of samples to load

    Returns:
        List of samples with 'input', 'context', 'answers' fields
    """
    import json
    import os
    import urllib.request
    import zipfile

    try:
        cache_dir = os.path.expanduser("~/.cache/longbench")
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "data.zip")
        extract_dir = os.path.join(cache_dir, "data")

        # Download zip if not extracted yet
        if not os.path.exists(extract_dir):
            if not os.path.exists(zip_path):
                url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
                logger.info(f"Downloading LongBench data.zip (~114MB)...")
                urllib.request.urlretrieve(url, zip_path)

            # Extract
            logger.info("Extracting data.zip...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(cache_dir)

        # Find the jsonl file
        jsonl_path = os.path.join(extract_dir, f"{task_name}.jsonl")
        if not os.path.exists(jsonl_path):
            # List available files for debugging
            available = [f for f in os.listdir(extract_dir) if f.endswith(".jsonl")] if os.path.exists(extract_dir) else []
            logger.error(f"Task {task_name}.jsonl not found. Available: {available[:10]}...")
            return []

        # Load JSONL
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        dataset = data

        samples = []
        for item in dataset:
            sample = {
                "input": item.get("input", ""),
                "context": item.get("context", ""),
                "answers": item.get("answers", []),
            }
            # Ensure answers is a list
            if isinstance(sample["answers"], str):
                sample["answers"] = [sample["answers"]]

            samples.append(sample)

            if num_samples and len(samples) >= num_samples:
                break

        logger.info(f"Loaded {len(samples)} samples for {task_name}")
        return samples

    except Exception as e:
        logger.error(f"Failed to load task {task_name}: {e}")
        return []


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
    elif method_name == "snapkv":
        config = SnapKVConfig(num_sink=num_sink, num_recent=num_recent, budget=budget)
        method = SnapKV(config)
    elif method_name == "pyramidkv":
        config = PyramidKVConfig(num_sink=num_sink, num_recent=num_recent, budget=budget)
        method = PyramidKV(config)
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


def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    """
    Compute token-level F1 score (SQuAD/LongBench standard).

    Uses Counter intersection to properly count duplicate tokens.
    """
    from collections import Counter

    pred_tokens = normalize_answer(prediction).split()

    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue

        # Use Counter intersection to count duplicate tokens correctly
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def compute_rouge_l(prediction: str, ground_truths: list[str]) -> float:
    """Compute ROUGE-L score."""

    def lcs_length(x: list[str], y: list[str]) -> int:
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


def compute_accuracy(prediction: str, ground_truths: list[str]) -> float:
    """Compute exact match accuracy."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == pred_norm:
            return 1.0
    return 0.0


def compute_edit_similarity(prediction: str, ground_truths: list[str]) -> float:
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
    sample: dict,
    method: str,
    sidecar: SidecarPPL | None,
    task_config: dict,
    window_size: int,
    num_sink: int,
    num_recent: int,
    max_context_length: int,
) -> tuple[float, str]:
    """
    Evaluate a single sample.

    Returns:
        Tuple of (score, generated_text)
    """
    device = next(model.parameters()).device

    # Build prompt with proper truncation that preserves the question
    context = sample["context"]
    question = sample["input"]

    # Tokenize question suffix first to reserve space for it
    question_suffix = f"\n\nQuestion: {question}\nAnswer:"
    question_tokens = tokenizer(question_suffix, add_special_tokens=False, return_tensors="pt")
    question_len = question_tokens.input_ids.shape[1]

    # Reserve tokens for question (with some buffer for generation)
    context_budget = max_context_length - question_len - 10

    # Tokenize and truncate context only
    context_tokens = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=context_budget,
        add_special_tokens=True,  # Include BOS token
    )

    # Concatenate: [context_tokens] + [question_tokens]
    input_ids = torch.cat(
        [context_tokens.input_ids, question_tokens.input_ids], dim=1
    ).to(device)

    # Get cache from context (all but last few tokens)
    context_len = max(1, input_ids.shape[1] - 20)
    context_ids = input_ids[:, :context_len]
    question_ids = input_ids[:, context_len:]

    with torch.no_grad():
        outputs = model(context_ids, use_cache=True)
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
    budget = num_sink + num_recent + (context_len - num_sink - num_recent) // window_size

    # Compress cache
    if method == "dense":
        compressed = cache
    elif method == "lst":
        compressed = compress_cache_lst(cache, sidecar, window_size, num_sink, num_recent)
    elif method == "mean":
        compressed = compress_cache_mean(cache, window_size, num_sink, num_recent)
    else:
        compressed = compress_cache_with_baseline(cache, method, budget, num_sink, num_recent)

    # Generate with compressed cache using manual autoregressive generation
    from transformers.cache_utils import DynamicCache

    max_new_tokens = task_config.get("max_gen", 64)

    # Convert cache list to DynamicCache for newer transformers
    past_kv = DynamicCache()
    for k, v in compressed:
        past_kv.update(k, v, layer_idx=len(past_kv))

    # Get cache length from the actual compressed cache
    cache_len = compressed[0][0].shape[2] if compressed else 0

    # Manual autoregressive generation (works better with external cache)
    all_tokens = question_ids.clone()

    with torch.no_grad():
        # Process all question tokens in ONE forward pass (not token-by-token)
        q_len = question_ids.shape[1]
        position_ids = torch.arange(cache_len, cache_len + q_len, device=device).unsqueeze(0)
        out = model(
            question_ids,
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
        all_tokens[0, question_ids.shape[1] :],
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
    sidecar: SidecarPPL | None,
    window_size: int,
    num_sink: int,
    num_recent: int,
    num_samples: int,
    max_context_length: int,
) -> dict:
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
                model,
                tokenizer,
                sample,
                method,
                sidecar,
                task_config,
                window_size,
                num_sink,
                num_recent,
                max_context_length,
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
        default="dense,lst,mean,h2o,streaming,tova,snapkv,pyramidkv",
        help="Comma-separated list of methods (dense,lst,mean,h2o,streaming,tova,snapkv,pyramidkv,cam,weightedkv)",
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
            sidecar = load_sidecar(args.checkpoint, device, model.config)

    # Evaluate
    all_results = {}

    for method in methods:
        logger.info(f"Evaluating method: {method}")
        method_results = {}

        for task in tasks:
            logger.info(f"  Task: {task}")
            try:
                result = evaluate_task(
                    model,
                    tokenizer,
                    task,
                    method,
                    sidecar,
                    args.window_size,
                    args.num_sink,
                    args.num_recent,
                    args.num_samples,
                    args.max_context_length,
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
