#!/usr/bin/env python3
"""
LongBench Evaluation Script
============================

Evaluate LST and baselines on LongBench, a comprehensive benchmark
for long-context language understanding.

This implementation follows the official LongBench evaluation protocol:
- Uses official prompt templates from THUDM/LongBench
- Applies task-specific prediction preprocessing
- Uses standard metrics (F1, ROUGE-L, classification score, etc.)
- Reports scores as percentages (0-100) matching published results

Dependencies:
    pip install rouge fuzzywuzzy python-Levenshtein

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
        - trec: Question classification (uses classification metric)
        - triviaqa: Trivia questions
        - samsum: Dialogue summarization

    Synthetic:
        - passage_count: Count unique paragraphs (uses count metric)
        - passage_retrieval_en: Retrieve relevant passage (uses retrieval metric)

    Code:
        - lcc: Long code completion (uses code_sim metric)
        - repobench-p: Repository-level code completion (uses code_sim metric)

Reference:
    - Bai et al. "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
    - https://github.com/THUDM/LongBench
    - https://github.com/Zefan-Cai/KVCache-Factory
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

# Task configurations with updated metrics to match official LongBench
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
    "trec": {"type": "classification", "metric": "classification", "max_gen": 16},
    "triviaqa": {"type": "qa", "metric": "f1", "max_gen": 32},
    "samsum": {"type": "summarization", "metric": "rouge", "max_gen": 128},
    # Synthetic - use official count/retrieval metrics
    "passage_count": {"type": "counting", "metric": "count", "max_gen": 32},
    "passage_retrieval_en": {"type": "retrieval", "metric": "retrieval", "max_gen": 32},
    # Code
    "lcc": {"type": "code", "metric": "code_sim", "max_gen": 64},
    "repobench-p": {"type": "code", "metric": "code_sim", "max_gen": 64},
}

# TREC classification labels (fine-grained, from official LongBench data)
# These are the actual labels used in the few-shot examples
TREC_CLASSES = [
    # Fine-grained classes from LongBench TREC data
    "Abbreviation", "Expression abbreviated",
    "Animal", "Body of water", "Color", "Currency name", "Disease and medicine",
    "Element and substance", "Event", "Food", "Instrument", "Language",
    "Letter like a-z", "Musical instrument", "Other entity", "Plant", "Product",
    "Religion", "Sport", "Symbols and sign", "Techniques and method", "Vehicle",
    "Word with a special property",
    "Definition of something", "Description of a person", "Description of something",
    "Equivalent term", "Manner of an action", "Reason",
    "Group or organization of person", "Individual", "Title of a person",
    "City", "Country", "Mountain", "Other location", "State",
    "Date", "Distance, linear measure", "Lasting time of somethin", "Money",
    "Number of something", "Order, rank", "Other number", "Percent, fraction",
    "Postcode or other code", "Price", "Size, area and volume", "Speed",
    "Temperature", "Weight", "Invention, book and other creative piece",
    "Organ of body",
]

# Prompt templates - EXACT match of KVCache-Factory's run_longbench.py
# Reference: https://github.com/Zefan-Cai/KVCache-Factory/blob/main/run_longbench.py
DATASET_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question asconcisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\nNow, answer the question based on the story as"
        "concisely as you can, using a single phrase if possible. Do not provide "
        "any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n Answer the question based on the above article as "
        "concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\nNow, answer the following question based on the above text, "
        "only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary "
        "of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or "
        "instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or "
        "more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer "
        "and do not output any other words. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are "
        "some examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may "
        "be duplicates. Please carefully read these paragraphs and determine how "
        "many unique paragraphs there are after removing duplicates. In other words, "
        "how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
        "The answer is: "
    ),
    "lcc": (
        "Please complete the code given below. \n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line of code:\n"
    ),
}

DEFAULT_TASKS = ["narrativeqa", "hotpotqa", "gov_report", "trec", "passage_count"]

# Model-specific max context lengths (from KVCache-Factory)
# These are critical for proper LongBench evaluation!
MODEL_MAX_LENGTHS = {
    "llama-2": 3950,
    "llama-3": 7950,
    "llama-3.1": 127500,  # 128k context
    "mistral": 31500,  # 32k context
    "mixtral": 31500,
    "qwen": 31500,
    "yi": 199500,  # 200k context
    "phi": 127500,
    "gemma": 7950,
    "tinyllama": 2048,
    # Default fallback
    "default": 4096,
}


def get_model_max_length(model_name: str) -> int:
    """
    Get the appropriate max context length for a model.

    Uses model-specific limits from KVCache-Factory for accurate benchmarking.
    """
    model_name_lower = model_name.lower()

    # Check for specific model patterns
    if "llama-3.1" in model_name_lower or "llama3.1" in model_name_lower:
        return MODEL_MAX_LENGTHS["llama-3.1"]
    elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
        return MODEL_MAX_LENGTHS["llama-3"]
    elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
        return MODEL_MAX_LENGTHS["llama-2"]
    elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
        return MODEL_MAX_LENGTHS["mistral"]
    elif "qwen" in model_name_lower:
        return MODEL_MAX_LENGTHS["qwen"]
    elif "yi" in model_name_lower:
        return MODEL_MAX_LENGTHS["yi"]
    elif "phi" in model_name_lower:
        return MODEL_MAX_LENGTHS["phi"]
    elif "gemma" in model_name_lower:
        return MODEL_MAX_LENGTHS["gemma"]
    elif "tinyllama" in model_name_lower:
        return MODEL_MAX_LENGTHS["tinyllama"]

    return MODEL_MAX_LENGTHS["default"]


def build_chat(tokenizer, prompt: str, model_name: str) -> str:
    """
    Build chat prompt following KVCache-Factory's EXACT approach.

    IMPORTANT: KVCache-Factory only applies [INST] wrapper for llama2 models,
    NOT for Mistral or other models. This matches their run_longbench.py exactly.

    Reference: https://github.com/Zefan-Cai/KVCache-Factory/blob/main/run_longbench.py
    """
    model_name_lower = model_name.lower()

    # KVCache-Factory ONLY applies chat format for llama2
    if "llama2" in model_name_lower or "llama-2" in model_name_lower:
        return f"[INST] {prompt} [/INST]"

    # For all other models (including Mistral), return raw prompt
    return prompt


def truncate_to_max_length(
    tokenizer,
    prompt: str,
    max_length: int,
) -> str:
    """
    Truncate prompt to max_length using KVCache-Factory's exact approach.

    If the tokenized prompt exceeds max_length, takes the first half and
    last half of tokens, decodes them, and concatenates.

    This preserves the beginning (instructions) and end (question) of the prompt.

    Reference: https://github.com/Zefan-Cai/KVCache-Factory/blob/main/run_longbench.py
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)

    if len(input_ids) <= max_length:
        return prompt

    # KVCache-Factory's exact truncation: first half + last half
    half = max_length // 2
    first_half = tokenizer.decode(input_ids[:half], skip_special_tokens=True)
    last_half = tokenizer.decode(input_ids[-half:], skip_special_tokens=True)

    return first_half + last_half


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
                "all_classes": item.get("all_classes", []),  # For classification tasks
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
    num_sink: int,
    num_recent: int,
    compression_ratio: float = 2.0,
    max_capacity_prompt: int = 2048,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Generic cache compression using baseline methods.

    Args:
        cache: List of (key, value) tuples per layer
        method_name: Name of compression method
        num_sink: Number of sink tokens to preserve
        num_recent: Number of recent tokens to preserve
        compression_ratio: Compression ratio for ratio-based methods
        max_capacity_prompt: Max KV cache capacity for PyramidKV (official default: 2048)

    Notes:
        - PyramidKV uses max_capacity_prompt (official parameter) not compression_ratio
        - Other methods use compression_ratio for fair comparison with LST
    """
    num_layers = len(cache)

    # Create method instance based on method name
    if method_name == "h2o":
        config = H2OConfig(num_sink=num_sink, num_recent=num_recent, compression_ratio=compression_ratio)
        method = H2O(config)
    elif method_name == "streaming":
        config = StreamingLLMConfig(num_sink=num_sink, num_recent=num_recent, compression_ratio=compression_ratio)
        method = StreamingLLM(config)
    elif method_name == "tova":
        config = TOVAConfig(num_sink=num_sink, num_recent=num_recent, compression_ratio=compression_ratio)
        method = TOVA(config)
    elif method_name == "kvmerger":
        config = KVMergerConfig(num_sink=num_sink, num_recent=num_recent, compression_ratio=compression_ratio)
        method = KVMerger(config)
    elif method_name == "weightedkv":
        config = WeightedKVConfig(num_sink=num_sink, num_recent=num_recent, compression_ratio=compression_ratio)
        method = WeightedKV(config)
    elif method_name == "cam":
        config = CaMConfig(num_sink=num_sink, num_recent=num_recent, compression_ratio=compression_ratio)
        method = CaM(config)
    elif method_name == "snapkv":
        config = SnapKVConfig(num_sink=num_sink, num_recent=num_recent, compression_ratio=compression_ratio)
        method = SnapKV(config)
    elif method_name == "pyramidkv":
        # PyramidKV uses official parameters from the paper
        config = PyramidKVConfig(
            num_sink=num_sink,
            num_recent=num_recent,
            num_layers=num_layers,
            max_capacity_prompt=max_capacity_prompt,
            window_size=32,  # Official default
            beta=20,  # Official default
        )
        method = PyramidKV(config)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    compressed = []

    for layer_idx, (k, v) in enumerate(cache):
        B, H, S, D = k.shape

        k_flat = k.transpose(1, 2).reshape(B, S, H * D)
        v_flat = v.transpose(1, 2).reshape(B, S, H * D)

        # Pass layer_idx for layer-aware methods like PyramidKV
        k_comp, v_comp = method.compress(k_flat, v_flat, layer_idx=layer_idx)

        S_new = k_comp.shape[1]
        k_comp = k_comp.view(B, S_new, H, D).transpose(1, 2)
        v_comp = v_comp.view(B, S_new, H, D).transpose(1, 2)

        compressed.append((k_comp, v_comp))

    return compressed


def preprocess_prediction(prediction: str, task_name: str) -> str:
    """
    Preprocess prediction based on task type (official LongBench standard).

    For classification/short-answer tasks, take only the first line
    to avoid penalizing models that provide explanations.
    """
    # Tasks that require first-line extraction (per official LongBench eval.py)
    first_line_tasks = {"trec", "triviaqa", "samsum", "lsht"}

    if task_name in first_line_tasks:
        prediction = prediction.lstrip('\n').split('\n')[0]

    return prediction


def normalize_answer(text: str) -> str:
    """
    Normalize answer for comparison (official LongBench standard).

    Order: lowercase -> remove punctuation -> remove articles -> fix whitespace
    """
    import re
    import string

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


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
    """
    Compute ROUGE-L score using the rouge library (official LongBench standard).

    Falls back to custom implementation if rouge library fails.
    """
    try:
        from rouge import Rouge
        rouge = Rouge()
    except ImportError:
        logger.warning("rouge library not installed, using fallback implementation")
        return _compute_rouge_l_fallback(prediction, ground_truths)

    if not prediction.strip():
        return 0.0

    best_score = 0.0
    for gt in ground_truths:
        if not gt.strip():
            continue
        try:
            scores = rouge.get_scores([prediction], [gt], avg=True)
            score = scores["rouge-l"]["f"]
            best_score = max(best_score, score)
        except Exception:
            # Fallback for edge cases (empty strings, etc.)
            continue

    return best_score


def _compute_rouge_l_fallback(prediction: str, ground_truths: list[str]) -> float:
    """Fallback ROUGE-L implementation if rouge library unavailable."""

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


def classification_score(prediction: str, ground_truth: str, all_classes: list[str]) -> float:
    """
    Compute classification score - exact match of KVCache-Factory's metrics.py.

    Reference: https://github.com/Zefan-Cai/KVCache-Factory/blob/main/metrics.py

    Args:
        prediction: Model's output (single string)
        ground_truth: Single correct answer (string, not list)
        all_classes: List of all possible class labels
    """
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:  # Case-sensitive, exact match of their code
            em_match_list.append(class_name)

    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)

    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0

    return score


def compute_classification_score(
    prediction: str, ground_truths: list[str], all_classes: list[str] | None = None
) -> float:
    """
    Wrapper that matches KVCache-Factory's eval.py calling pattern.

    They iterate through ground_truths and take the MAX score.
    """
    if all_classes is None or not all_classes:
        return _compute_accuracy_fallback(prediction, ground_truths)

    # Match their eval.py: iterate through ground_truths and take max
    max_score = 0.0
    for ground_truth in ground_truths:
        score = classification_score(prediction, ground_truth, all_classes)
        max_score = max(max_score, score)

    return max_score


def _compute_accuracy_fallback(prediction: str, ground_truths: list[str]) -> float:
    """Fallback simple accuracy when all_classes not available."""
    pred_norm = normalize_answer(prediction)

    for gt in ground_truths:
        gt_norm = normalize_answer(gt)

        # Exact match
        if gt_norm == pred_norm:
            return 1.0

        # Check if ground truth label is contained in prediction
        if gt_norm in pred_norm:
            return 1.0

    return 0.0


def count_score(prediction: str, ground_truth: str) -> float:
    """
    Count score - exact match of KVCache-Factory's metrics.py.

    Reference: https://github.com/Zefan-Cai/KVCache-Factory/blob/main/metrics.py
    """
    import re

    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def compute_count_score(prediction: str, ground_truths: list[str]) -> float:
    """Wrapper that iterates through ground_truths and takes max (matches eval.py)."""
    max_score = 0.0
    for ground_truth in ground_truths:
        score = count_score(prediction, ground_truth)
        max_score = max(max_score, score)
    return max_score


def retrieval_score(prediction: str, ground_truth: str) -> float:
    """
    Retrieval score - exact match of KVCache-Factory's metrics.py.

    Reference: https://github.com/Zefan-Cai/KVCache-Factory/blob/main/metrics.py
    """
    import re

    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]

    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def compute_retrieval_score(prediction: str, ground_truths: list[str]) -> float:
    """Wrapper that iterates through ground_truths and takes max (matches eval.py)."""
    max_score = 0.0
    for ground_truth in ground_truths:
        score = retrieval_score(prediction, ground_truth)
        max_score = max(max_score, score)
    return max_score


def code_sim_score(prediction: str, ground_truth: str) -> float:
    """
    Code similarity score - exact match of KVCache-Factory's metrics.py.

    Reference: https://github.com/Zefan-Cai/KVCache-Factory/blob/main/metrics.py
    """
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        return 0.0

    all_lines = prediction.lstrip('\n').split('\n')
    processed_prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            processed_prediction = line
            break

    return fuzz.ratio(processed_prediction, ground_truth) / 100


def compute_code_similarity(prediction: str, ground_truths: list[str]) -> float:
    """Wrapper that iterates through ground_truths and takes max (matches eval.py)."""
    max_score = 0.0
    for ground_truth in ground_truths:
        score = code_sim_score(prediction, ground_truth)
        max_score = max(max_score, score)
    return max_score


def _compute_edit_similarity_fallback(prediction: str, ground_truths: list[str]) -> float:
    """Fallback edit similarity if fuzzywuzzy unavailable."""

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

    # Apply same preprocessing as official
    all_lines = prediction.lstrip('\n').split('\n')
    processed_prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            processed_prediction = line
            break

    if not processed_prediction:
        processed_prediction = prediction.lstrip('\n').split('\n')[0] if prediction.strip() else ""

    best_sim = 0.0
    for gt in ground_truths:
        max_len = max(len(processed_prediction), len(gt))
        if max_len == 0:
            continue
        dist = edit_distance(processed_prediction, gt)
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
    task_name: str,
    model_name: str,
    window_size: int,
    num_sink: int,
    num_recent: int,
    max_context_length: int,
    max_capacity_prompt: int = 2048,
    debug: bool = False,
) -> tuple[float, str]:
    """
    Evaluate a single sample.

    Args:
        model: The language model
        tokenizer: The tokenizer
        sample: Sample dict with 'input', 'context', 'answers'
        method: Compression method name
        sidecar: LST sidecar model (if method='lst')
        task_config: Task configuration dict
        task_name: Name of the LongBench task (for prompt selection)
        model_name: Model name (for chat template selection)
        window_size: Compression window size
        num_sink: Number of sink tokens
        num_recent: Number of recent tokens
        max_context_length: Maximum context length
        max_capacity_prompt: Max KV cache capacity for PyramidKV
        debug: If True, print debug information

    Returns:
        Tuple of (score, generated_text)
    """
    device = next(model.parameters()).device
    max_new_tokens = task_config.get("max_gen", 64)

    # Build prompt using KVCache-Factory's exact approach
    context = sample["context"]
    question = sample["input"]

    # Get prompt template for this task
    prompt_template = DATASET_PROMPTS.get(task_name)
    if not prompt_template:
        prompt_template = "{context}\n\nQuestion: {input}\nAnswer:"

    # Step 1: Build prompt from template
    prompt = prompt_template.format(context=context, input=question)

    # Step 2: Apply chat template for instruction models
    prompt = build_chat(tokenizer, prompt, model_name)

    # Step 3: Truncate if needed (KVCache-Factory's first-half + last-half approach)
    prompt = truncate_to_max_length(tokenizer, prompt, max_context_length)

    # Step 4: Tokenize
    tokenized = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    prompt_len = input_ids.shape[1]

    if debug:
        logger.info(f"Prompt length: {prompt_len} tokens")
        logger.info(f"Prompt (last 200 chars): ...{prompt[-200:]}")

    # For dense (no compression), use model.generate() directly
    # Parameters match KVCache-Factory's run_longbench.py exactly
    if method == "dense":
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=prompt_len + 1,  # KVCache-Factory: ensures at least 1 token generated
                eos_token_id=[tokenizer.eos_token_id],  # KVCache-Factory uses list
            )
        # Decode only the generated part
        response = tokenizer.decode(
            output_ids[0, prompt_len:],
            skip_special_tokens=True,
        ).strip()
    else:
        # For compression methods, use cache-based generation
        # Split: process most tokens to build cache, keep last few for generation
        split_point = max(1, prompt_len - 20)
        context_ids = input_ids[:, :split_point]
        question_ids = input_ids[:, split_point:]

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

        # Compress cache based on method
        if method == "lst":
            compressed = compress_cache_lst(cache, sidecar, window_size, num_sink, num_recent)
        elif method == "mean":
            compressed = compress_cache_mean(cache, window_size, num_sink, num_recent)
        else:
            compressed = compress_cache_with_baseline(
                cache, method, num_sink, num_recent,
                compression_ratio=float(window_size),
                max_capacity_prompt=max_capacity_prompt,
            )

        # Generate with compressed cache
        from transformers.cache_utils import DynamicCache

        past_kv = DynamicCache()
        for k, v in compressed:
            past_kv.update(k, v, layer_idx=len(past_kv))

        cache_len = compressed[0][0].shape[2] if compressed else 0
        all_tokens = question_ids.clone()

        with torch.no_grad():
            # Process remaining question tokens
            q_len = question_ids.shape[1]
            position_ids = torch.arange(
                cache_len, cache_len + q_len, device=device, dtype=torch.long
            ).unsqueeze(0)

            out = model(
                question_ids,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
            )
            past_kv = out.past_key_values

            # Generate new tokens
            for _ in range(max_new_tokens):
                pos = cache_len + all_tokens.shape[1]
                position_ids = torch.tensor([[pos]], device=device, dtype=torch.long)
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

        response = tokenizer.decode(
            all_tokens[0, question_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

    # Apply task-specific preprocessing (official LongBench standard)
    processed_response = preprocess_prediction(response, task_name)

    if debug:
        logger.info(f"Raw response: {response[:200]}...")
        logger.info(f"Processed response: {processed_response[:100]}")
        logger.info(f"Expected answers: {sample.get('answers', [])[:3]}")

    # Compute score based on metric
    answers = sample.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]

    metric = task_config.get("metric", "f1")
    if metric == "f1":
        score = compute_f1(processed_response, answers)
    elif metric == "rouge":
        score = compute_rouge_l(processed_response, answers)
    elif metric == "classification":
        # Use all_classes from sample data, fall back to TREC_CLASSES
        all_classes = sample.get("all_classes", [])
        if not all_classes:
            all_classes = TREC_CLASSES if task_name == "trec" else None
        score = compute_classification_score(processed_response, answers, all_classes)
        if debug:
            # Show which classes matched
            matched = [c for c in (all_classes or []) if c in processed_response]
            logger.info(f"Matched classes: {matched}, Score: {score}")
    elif metric == "count":
        score = compute_count_score(processed_response, answers)
    elif metric == "retrieval":
        score = compute_retrieval_score(processed_response, answers)
    elif metric == "code_sim":
        score = compute_code_similarity(processed_response, answers)
    else:
        score = 0.0

    return score, response


def evaluate_task(
    model,
    tokenizer,
    task_name: str,
    method: str,
    model_name: str,
    sidecar: SidecarPPL | None,
    window_size: int,
    num_sink: int,
    num_recent: int,
    num_samples: int,
    max_context_length: int,
    max_capacity_prompt: int = 2048,
    debug: bool = False,
) -> dict:
    """Evaluate a single LongBench task."""
    task_config = TASK_CONFIGS.get(task_name, {"type": "qa", "metric": "f1", "max_gen": 64})

    # Load data
    samples = load_longbench_task(task_name, num_samples)
    if not samples:
        return {"score": 0.0, "n_samples": 0, "error": "Failed to load data"}

    scores = []
    for idx, sample in enumerate(tqdm(samples, desc=f"{task_name}", leave=False)):
        # Enable debug only for first 3 samples when debug flag is set
        sample_debug = debug and idx < 3
        try:
            score, _ = evaluate_sample(
                model,
                tokenizer,
                sample,
                method,
                sidecar,
                task_config,
                task_name,
                model_name,
                window_size,
                num_sink,
                num_recent,
                max_context_length,
                max_capacity_prompt=max_capacity_prompt,
                debug=sample_debug,
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
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=None,
        help="Max context length (auto-detected from model if not specified)",
    )
    parser.add_argument("--window_size", type=int, default=8, help="Compression window size for LST/mean")
    parser.add_argument("--num_sink", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--num_recent", type=int, default=8, help="Number of recent tokens")
    parser.add_argument(
        "--max_capacity_prompt",
        type=int,
        default=2048,
        help="Max KV cache capacity for PyramidKV (official default: 2048)",
    )
    parser.add_argument("--output_file", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for first few samples")

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

    # Determine max context length (auto-detect from model if not specified)
    if args.max_context_length is None:
        max_context_length = get_model_max_length(args.model_name)
        logger.info(f"Auto-detected max_context_length: {max_context_length} for {args.model_name}")
    else:
        max_context_length = args.max_context_length
        logger.info(f"Using specified max_context_length: {max_context_length}")

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
                    args.model_name,
                    sidecar,
                    args.window_size,
                    args.num_sink,
                    args.num_recent,
                    args.num_samples,
                    max_context_length,
                    max_capacity_prompt=args.max_capacity_prompt,
                    debug=args.debug,
                )
                method_results[task] = result
                logger.info(f"    Score: {result['score']*100:.2f}% ({result['metric']})")
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                method_results[task] = {"score": 0.0, "error": str(e)}

        # Compute average
        scores = [r["score"] for r in method_results.values() if "score" in r]
        method_results["average"] = np.mean(scores) if scores else 0.0

        all_results[method] = method_results

    # Print summary (scores scaled to 0-100 per LongBench standard)
    print("\n" + "=" * 80)
    print("LONGBENCH EVALUATION RESULTS (scores in %)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Max context length: {max_context_length}")
    print(f"Tasks: {tasks}")
    print(f"Samples per task: {args.num_samples}")
    print(f"Window size: {args.window_size} ({args.window_size}:1 compression)")
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
            # Scale score to percentage (0-100) per LongBench standard
            score = results.get(task, {}).get("score", 0) * 100
            row += f" {score:>12.2f}"
        avg = results.get("average", 0) * 100
        row += f" {avg:>10.2f}"
        print(row)

    print("=" * 80)

    # Save results (scale scores to percentages for consistency)
    if args.output_file:
        scaled_results = {}
        for method, results in all_results.items():
            scaled_results[method] = {}
            for key, val in results.items():
                if isinstance(val, dict) and "score" in val:
                    scaled_results[method][key] = {
                        **val,
                        "score": round(val["score"] * 100, 2)
                    }
                elif key == "average":
                    scaled_results[method][key] = round(val * 100, 2)
                else:
                    scaled_results[method][key] = val

        output = {
            "model": args.model_name,
            "tasks": tasks,
            "methods": methods,
            "results": scaled_results,
            "note": "Scores are in percentage (0-100) per LongBench standard",
        }
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
