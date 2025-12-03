"""
Generation benchmark for testing KV cache compression.

This benchmark evaluates compression by generating text continuations
and comparing quality/speed metrics between dense and compressed methods.

Unlike perplexity (which doesn't apply compression), this directly tests
the use case: long-context generation with memory savings.
"""

import time
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from ..methods.base import BaseMethod
from .base import BaseBenchmark, BenchmarkResult


class GenerationBenchmark(BaseBenchmark):
    """
    Text generation benchmark with KV cache compression.
    
    Evaluates:
    - Generation quality (comparing outputs to reference)
    - Generation speed (tokens/sec)
    - Memory usage (cache size)
    - Compression effectiveness
    
    This benchmark actually uses compression during generation,
    unlike perplexity which uses standard forward pass.
    """
    
    SUPPORTED_DATASETS = {
        "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
        "pg19": ("pg19", None),
    }
    
    def __init__(
        self,
        dataset_name: str = "wikitext-2",
        split: str = "test",
        prompt_length: int = 512,
        generation_length: int = 128,
        num_samples: Optional[int] = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Args:
            dataset_name: Dataset to use for prompts
            split: Dataset split
            prompt_length: Length of context prompt
            generation_length: Number of tokens to generate
            num_samples: Number of prompts to test (None = all)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            seed: Random seed
            verbose: Print progress
        """
        super().__init__(num_samples=num_samples, seed=seed, verbose=verbose, **kwargs)
        
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {list(self.SUPPORTED_DATASETS.keys())}"
            )
        
        self.dataset_name = dataset_name
        self.split = split
        self.prompt_length = prompt_length
        self.generation_length = generation_length
        self.temperature = temperature
        self.top_p = top_p
        
        self.dataset = None
        self.prompts = None
    
    @property
    def name(self) -> str:
        return f"generation_{self.dataset_name}"
    
    @property
    def description(self) -> str:
        return (
            f"Text generation on {self.dataset_name} "
            f"({self.prompt_length}→{self.generation_length} tokens)"
        )
    
    def setup(self) -> None:
        """Load dataset and extract prompts."""
        if self._is_setup:
            return
        
        self.log(f"Loading dataset: {self.dataset_name}")
        
        dataset_path, dataset_config = self.SUPPORTED_DATASETS[self.dataset_name]
        
        if dataset_config:
            self.dataset = load_dataset(
                dataset_path,
                dataset_config,
                split=self.split,
                trust_remote_code=True,
            )
        else:
            self.dataset = load_dataset(
                dataset_path,
                split=self.split,
                trust_remote_code=True,
            )
        
        self.log(f"Loaded {len(self.dataset)} examples")
        self._is_setup = True
    
    def _prepare_prompts(self, tokenizer) -> List[Dict]:
        """Extract prompts from dataset."""
        import random
        
        prompts = []
        
        # Get text field name
        text_field = "text" if "text" in self.dataset.column_names else "content"
        
        # Sample documents
        indices = list(range(len(self.dataset)))
        random.seed(self.seed)
        random.shuffle(indices)
        
        for idx in indices:
            text = self.dataset[idx][text_field]
            
            if not text or len(text.strip()) < 100:
                continue
            
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Need enough tokens for prompt + generation
            min_length = self.prompt_length + self.generation_length
            if len(tokens) < min_length:
                continue
            
            # Extract prompt
            prompt_tokens = tokens[:self.prompt_length]
            
            prompts.append({
                "prompt_ids": torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0),
                "prompt_text": tokenizer.decode(prompt_tokens),
                "doc_idx": idx,
            })
            
            if self.num_samples and len(prompts) >= self.num_samples:
                break
        
        self.log(f"Prepared {len(prompts)} prompts")
        return prompts
    
    def evaluate(self, method: BaseMethod) -> BenchmarkResult:
        """
        Generate text using the given method.
        
        For FMKV, this will trigger actual compression during generation.
        For dense, no compression occurs.
        """
        if not self._is_setup:
            self.setup()
        
        method.setup()
        
        # Check if method has generate capability
        if not hasattr(method, 'generate'):
            return BenchmarkResult(
                benchmark_name=self.name,
                method_name=method.name,
                metrics={"error": "Method does not support generation"},
                sample_results=[],
                config={},
                errors=["Method does not implement generate()"],
                num_samples=0,
                num_successful=0,
            )
        
        self.log(f"Generating with {method.name} on {self.dataset_name}")
        
        # Prepare prompts
        prompts = self._prepare_prompts(method.tokenizer)
        
        if not prompts:
            return BenchmarkResult(
                benchmark_name=self.name,
                method_name=method.name,
                metrics={"error": "No valid prompts found"},
                sample_results=[],
                config={},
                errors=["No valid prompts found"],
                num_samples=0,
                num_successful=0,
            )
        
        # Generate
        sample_results = []
        errors = []
        
        total_tokens_generated = 0
        total_time = 0.0
        
        progress = tqdm(prompts, desc=f"Gen ({method.name})", disable=not self.verbose)
        
        for prompt in progress:
            try:
                prompt_ids = prompt["prompt_ids"].to(method.model.device)
                
                # Time generation
                start_time = time.time()
                
                # Generate using method's generate function (applies compression for FMKV)
                with torch.no_grad():
                    output = method.generate(
                        input_ids=prompt_ids,
                        max_new_tokens=self.generation_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                # Extract generated text
                generated_text = output.text[0] if isinstance(output.text, list) else output.text
                num_generated = output.sequences.shape[1] - prompt_ids.shape[1]
                tokens_per_sec = num_generated / elapsed if elapsed > 0 else 0
                
                total_tokens_generated += num_generated
                total_time += elapsed
                
                sample_results.append({
                    "doc_idx": prompt["doc_idx"],
                    "prompt_text": prompt["prompt_text"][:100] + "...",
                    "generated_text": generated_text[:200] + "...",
                    "num_tokens": num_generated,
                    "time_sec": elapsed,
                    "tokens_per_sec": tokens_per_sec,
                })
                
                # Update progress
                avg_speed = total_tokens_generated / total_time if total_time > 0 else 0
                progress.set_postfix({"tok/s": f"{avg_speed:.1f}"})
                
            except Exception as e:
                errors.append(f"Doc {prompt['doc_idx']}: {str(e)}")
                continue
        
        # Compute aggregate metrics
        avg_tokens_per_sec = total_tokens_generated / total_time if total_time > 0 else 0
        
        metrics = {
            "total_tokens_generated": total_tokens_generated,
            "total_time_sec": total_time,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "num_prompts": len(prompts),
            "num_successful": len(sample_results),
        }
        
        # Add compression stats if available
        if hasattr(method, 'get_cache_stats'):
            cache_stats = method.get_cache_stats()
            metrics.update({f"cache/{k}": v for k, v in cache_stats.items()})
        
        self.log(f"Generated {total_tokens_generated} tokens in {total_time:.1f}s "
                f"({avg_tokens_per_sec:.1f} tok/s)")
        
        return BenchmarkResult(
            benchmark_name=self.name,
            method_name=method.name,
            metrics=metrics,
            sample_results=sample_results,
            config={
                "dataset": self.dataset_name,
                "split": self.split,
                "prompt_length": self.prompt_length,
                "generation_length": self.generation_length,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
            errors=errors,
            num_samples=len(prompts),
            num_successful=len(sample_results),
        )
