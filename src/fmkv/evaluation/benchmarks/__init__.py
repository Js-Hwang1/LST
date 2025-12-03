"""
Benchmark implementations for KV cache compression evaluation.

Standard benchmarks:
- Perplexity: Language modeling quality (WikiText-2, WikiText-103, C4)
- Generation: Text generation with compression (Bug #13 fix)
- Passkey Retrieval: Long-context retrieval accuracy
- LongBench: Comprehensive long-context benchmark suite
- Needle-in-Haystack: Single fact retrieval in long context
- Force Validation: Explicit force-matching error measurement (Bug #10 fix)
- Memory/Latency: Efficiency metrics
"""

from .base import BaseBenchmark, BenchmarkResult
from .perplexity import PerplexityBenchmark
from .generation import GenerationBenchmark
from .passkey import PasskeyRetrievalBenchmark
from .needle import NeedleInHaystackBenchmark
from .longbench import LongBenchBenchmark
from .force_validation import ForceValidationBenchmark

__all__ = [
    "BaseBenchmark",
    "BenchmarkResult",
    "PerplexityBenchmark",
    "GenerationBenchmark",
    "PasskeyRetrievalBenchmark",
    "NeedleInHaystackBenchmark",
    "LongBenchBenchmark",
    "ForceValidationBenchmark",
    "get_benchmark",
    "list_benchmarks",
]

_BENCHMARK_REGISTRY = {
    "perplexity": PerplexityBenchmark,
    "generation": GenerationBenchmark,
    "passkey": PasskeyRetrievalBenchmark,
    "needle": NeedleInHaystackBenchmark,
    "longbench": LongBenchBenchmark,
    "force_validation": ForceValidationBenchmark,
}


def get_benchmark(name: str, **kwargs) -> BaseBenchmark:
    """Get benchmark by name."""
    if name not in _BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark: {name}. Available: {list(_BENCHMARK_REGISTRY.keys())}"
        )
    return _BENCHMARK_REGISTRY[name](**kwargs)


def list_benchmarks() -> list[str]:
    """List available benchmarks."""
    return list(_BENCHMARK_REGISTRY.keys())

