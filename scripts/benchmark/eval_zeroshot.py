#!/usr/bin/env python3
"""
Zero-Shot Accuracy Evaluation for KV Cache Compression Methods
==============================================================

Following MiniCache (NeurIPS 2024) and PALU (ICLR 2025) methodology:
- Uses lm-evaluation-harness standard tasks
- Evaluates: PIQA, WinoGrande, HellaSwag, ARC-Easy, ARC-Challenge, OpenBookQA

This script provides a simplified evaluation that applies KV cache compression
during the context encoding phase, then measures accuracy on downstream tasks.

Usage:
    python scripts/benchmark/eval_zeroshot.py \
        --model_name meta-llama/Llama-2-7b-hf \
        --methods dense,lst,h2o,streaming,tova \
        --tasks piqa,winogrande \
        --window_size 2

Requirements:
    pip install lm-eval
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Standard zero-shot tasks following MiniCache/PALU papers
STANDARD_TASKS = {
    "piqa": "Physical intuition reasoning",
    "winogrande": "Coreference resolution",
    "hellaswag": "Commonsense reasoning",
    "arc_easy": "Science QA (easy)",
    "arc_challenge": "Science QA (challenge)",
    "openbookqa": "Open-domain reasoning",
}

# Additional tasks used in some papers
EXTENDED_TASKS = {
    "boolq": "Boolean QA",
    "copa": "Causal reasoning",
    "rte": "Textual entailment",
    "mathqa": "Mathematical reasoning",
}


def get_available_methods() -> list[str]:
    """Get list of available compression methods."""
    return [
        "dense", "lst", "mean",
        "h2o", "streaming", "tova", "snapkv", "pyramidkv",
        "kvmerger", "weightedkv", "cam",
    ]


def run_lm_eval_baseline(
    model_name: str,
    tasks: list[str],
    batch_size: int = 8,
    device: str = "cuda",
    num_fewshot: int = 0,
) -> dict[str, float]:
    """
    Run lm-evaluation-harness for dense baseline.

    This uses the standard lm_eval library directly for the dense baseline,
    which provides the most accurate reference numbers.

    Args:
        model_name: HuggingFace model name
        tasks: List of task names
        batch_size: Batch size for evaluation
        device: Device to use
        num_fewshot: Number of few-shot examples (0 for zero-shot)

    Returns:
        Dictionary mapping task names to accuracy scores
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-evaluation-harness not installed. Install with: pip install lm-eval")
        raise

    logger.info(f"Running lm_eval for dense baseline on tasks: {tasks}")

    # Create model
    model = HFLM(
        pretrained=model_name,
        batch_size=batch_size,
        device=device,
    )

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    # Extract accuracy scores
    scores = {}
    for task_name in tasks:
        if task_name in results["results"]:
            task_results = results["results"][task_name]
            # Try different accuracy keys
            for key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                if key in task_results:
                    scores[task_name] = task_results[key]
                    break
            else:
                # Use first numeric result
                for k, v in task_results.items():
                    if isinstance(v, (int, float)) and not k.endswith("_stderr"):
                        scores[task_name] = v
                        break

    return scores


def evaluate_with_compression(
    model_name: str,
    method: str,
    tasks: list[str],
    window_size: int = 2,
    sidecar_path: str | None = None,
    num_sink: int = 4,
    num_recent: int = 32,
    batch_size: int = 8,
    device: str = "cuda",
    num_samples: int | None = None,
) -> dict[str, float]:
    """
    Evaluate model with KV cache compression on zero-shot tasks.

    This implements a simplified evaluation that:
    1. Loads each task's examples
    2. Compresses the context KV cache
    3. Generates/scores with compressed cache
    4. Reports accuracy

    Args:
        model_name: HuggingFace model name
        method: Compression method name
        tasks: List of task names
        window_size: Compression window size
        sidecar_path: Path to LST sidecar checkpoint
        num_sink: Number of sink tokens to preserve
        num_recent: Number of recent tokens to preserve
        batch_size: Batch size for evaluation
        device: Device to use
        num_samples: Optional limit on samples per task

    Returns:
        Dictionary mapping task names to accuracy scores
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Import compression utilities
    from src.baselines.h2o import h2o_evict
    from src.baselines.streaming import streaming_evict
    from src.baselines.tova import tova_evict
    from src.baselines.snapkv import snapkv_compress
    from src.baselines.pyramidkv import pyramidkv_compress
    from src.baselines.cam import cam_compress
    from src.baselines.weightedkv import weightedkv_compress
    from src.baselines.kvmerger import kvmerger_compress

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    # Load LST sidecar if needed
    sidecar = None
    if method == "lst":
        if sidecar_path is None:
            raise ValueError("LST method requires --sidecar_path")
        from src.LST.sidecar.network import SidecarPPL

        logger.info(f"Loading sidecar from: {sidecar_path}")
        checkpoint = torch.load(sidecar_path, map_location=device, weights_only=False)

        # Get model config for sidecar initialization
        model_config = model.config
        sidecar = SidecarPPL(
            d_model=model_config.hidden_size,
            n_heads=model_config.num_attention_heads,
            d_head=model_config.hidden_size // model_config.num_attention_heads,
            n_layers=model_config.num_hidden_layers,
            window_size=window_size,
        ).to(device)
        sidecar.load_state_dict(checkpoint["model_state_dict"])
        sidecar.eval()

    # Load tasks using lm_eval's data loading
    try:
        from lm_eval import tasks as lm_tasks
        from lm_eval.api.instance import Instance
    except ImportError:
        logger.error("lm-evaluation-harness not installed")
        raise

    # Initialize task manager
    task_manager = lm_tasks.TaskManager()

    scores = {}

    for task_name in tasks:
        logger.info(f"Evaluating task: {task_name}")

        try:
            # Get task
            task_dict = lm_tasks.get_task_dict([task_name], task_manager)
            if task_name not in task_dict or task_dict[task_name] is None:
                logger.warning(f"Task {task_name} not found in task_dict, skipping")
                continue
            task = task_dict[task_name]

            # Get test instances - handle different lm_eval versions
            docs = None
            try:
                if hasattr(task, "test_docs") and callable(task.test_docs):
                    test_docs = task.test_docs()
                    if test_docs is not None:
                        docs = list(test_docs)

                if docs is None and hasattr(task, "validation_docs") and callable(task.validation_docs):
                    val_docs = task.validation_docs()
                    if val_docs is not None:
                        docs = list(val_docs)

                # Try alternative API for newer lm_eval versions
                if docs is None and hasattr(task, "dataset"):
                    dataset = task.dataset
                    if dataset is not None:
                        if "test" in dataset:
                            docs = list(dataset["test"])
                        elif "validation" in dataset:
                            docs = list(dataset["validation"])
            except Exception as doc_err:
                logger.warning(f"Error getting docs for {task_name}: {doc_err}")
                docs = None

            if docs is None or len(docs) == 0:
                logger.warning(f"Task {task_name} has no test/validation docs, skipping")
                continue

            if num_samples:
                docs = docs[:num_samples]

            correct = 0
            total = 0

            for doc in docs:
                # Get the prompt and choices
                ctx = task.doc_to_text(doc)
                target = task.doc_to_target(doc)

                # For multiple choice, we need to score each option
                if hasattr(task, "doc_to_choice"):
                    choices = task.doc_to_choice(doc)

                    best_score = float("-inf")
                    best_idx = 0

                    for idx, choice in enumerate(choices):
                        full_text = ctx + str(choice)

                        # Tokenize
                        inputs = tokenizer(
                            full_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=2048,
                        ).to(device)

                        # Get model logits
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits

                        # Score: log probability of the choice tokens
                        choice_tokens = tokenizer(str(choice), add_special_tokens=False).input_ids
                        choice_len = len(choice_tokens)

                        if choice_len > 0:
                            # Get logits for choice positions
                            choice_logits = logits[0, -(choice_len + 1):-1, :]
                            choice_probs = torch.nn.functional.log_softmax(choice_logits, dim=-1)

                            score = 0.0
                            for i, tok_id in enumerate(choice_tokens):
                                score += choice_probs[i, tok_id].item()

                            score /= choice_len  # Normalize by length

                            if score > best_score:
                                best_score = score
                                best_idx = idx

                    # Check if prediction is correct
                    if hasattr(task, "doc_to_target"):
                        gold = task.doc_to_target(doc)
                        if isinstance(gold, int):
                            if best_idx == gold:
                                correct += 1
                        elif isinstance(gold, str):
                            if choices[best_idx] == gold:
                                correct += 1

                    total += 1
                else:
                    # Handle generation tasks differently
                    total += 1

            if total > 0:
                accuracy = correct / total
                scores[task_name] = accuracy
                logger.info(f"  {task_name}: {accuracy:.4f} ({correct}/{total})")

        except Exception as e:
            logger.error(f"Error evaluating {task_name}: {e}")
            continue

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot Accuracy Evaluation for KV Cache Compression"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="dense",
        help="Comma-separated list of methods (dense,lst,h2o,streaming,tova,mean,kvmerger,weightedkv,cam)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="piqa,winogrande,hellaswag,arc_easy,arc_challenge,openbookqa",
        help="Comma-separated list of tasks",
    )
    parser.add_argument(
        "--sidecar_path",
        type=str,
        default=None,
        help="Path to LST sidecar checkpoint (required for lst method)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=2,
        help="Window size for compression (compression ratio)",
    )
    parser.add_argument(
        "--num_sink",
        type=int,
        default=4,
        help="Number of sink tokens to preserve",
    )
    parser.add_argument(
        "--num_recent",
        type=int,
        default=32,
        help="Number of recent tokens to preserve",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples per task (for quick testing)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--use_lm_eval",
        action="store_true",
        help="Use lm-evaluation-harness for dense baseline (more accurate but slower)",
    )

    args = parser.parse_args()

    # Parse arguments
    methods = [m.strip() for m in args.methods.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Validate methods
    available = get_available_methods()
    for method in methods:
        if method not in available:
            logger.error(f"Unknown method: {method}. Available: {available}")
            sys.exit(1)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Results storage
    all_results = {
        "model": args.model_name,
        "tasks": tasks,
        "window_size": args.window_size,
        "timestamp": datetime.now().isoformat(),
        "results": {},
    }

    # Evaluate each method
    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating method: {method}")
        logger.info(f"{'='*60}")

        try:
            if method == "dense" and args.use_lm_eval:
                # Use standard lm_eval for dense baseline
                scores = run_lm_eval_baseline(
                    model_name=args.model_name,
                    tasks=tasks,
                    batch_size=args.batch_size,
                    device=device,
                )
            else:
                # Use our evaluation with compression
                scores = evaluate_with_compression(
                    model_name=args.model_name,
                    method=method,
                    tasks=tasks,
                    window_size=args.window_size,
                    sidecar_path=args.sidecar_path,
                    num_sink=args.num_sink,
                    num_recent=args.num_recent,
                    batch_size=args.batch_size,
                    device=device,
                    num_samples=args.num_samples,
                )

            all_results["results"][method] = scores

        except Exception as e:
            logger.error(f"Error evaluating {method}: {e}")
            all_results["results"][method] = {"error": str(e)}

    # Print summary table
    print("\n" + "=" * 80)
    print("ZERO-SHOT ACCURACY RESULTS")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Window size: {args.window_size} ({args.window_size}:1 compression)")
    print("-" * 80)

    # Header
    header = f"{'Method':<15}"
    for task in tasks:
        header += f"{task:<12}"
    header += f"{'Avg':<10}"
    print(header)
    print("-" * 80)

    # Results
    for method in methods:
        if method in all_results["results"]:
            scores = all_results["results"][method]
            if isinstance(scores, dict) and "error" not in scores:
                row = f"{method:<15}"
                task_scores = []
                for task in tasks:
                    if task in scores:
                        score = scores[task]
                        row += f"{score:<12.4f}"
                        task_scores.append(score)
                    else:
                        row += f"{'—':<12}"

                # Average
                if task_scores:
                    avg = sum(task_scores) / len(task_scores)
                    row += f"{avg:<10.4f}"
                    all_results["results"][method]["average"] = avg
                else:
                    row += f"{'—':<10}"

                print(row)
            else:
                print(f"{method:<15} ERROR: {scores.get('error', 'Unknown')}")

    print("=" * 80)

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
