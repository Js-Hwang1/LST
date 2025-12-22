# KV Cache Compression Evaluation Guide

Comprehensive benchmark documentation for ICML 2026 submission, based on analysis of accepted papers at NeurIPS 2024, ICLR 2025, and other top venues.

---

## Table of Contents

1. [Perplexity Evaluation](#1-perplexity-evaluation)
2. [Zero-Shot Accuracy](#2-zero-shot-accuracy)
3. [LongBench](#3-longbench)
4. [Needle-in-a-Haystack (NIAH)](#4-needle-in-a-haystack-niah)
5. [RULER Benchmark](#5-ruler-benchmark)
6. [Efficiency Metrics](#6-efficiency-metrics)
7. [Model Selection Guide](#7-model-selection-guide)
8. [Baseline Implementations](#8-baseline-implementations)

---

## 1. Perplexity Evaluation

**Priority: CRITICAL** — Every KV cache compression paper uses perplexity as the primary quality metric.

### Datasets

| Dataset | Description | Sequence Length | Samples |
|---------|-------------|-----------------|---------|
| **WikiText-2** | Wikipedia articles (test set) | 1024 tokens | 2048 samples |
| **C4** | Common Crawl (validation set) | 1024 tokens | 2048 samples |

### Compression Ratios to Test

| Ratio | KV Budget | Description |
|-------|-----------|-------------|
| 2:1 | 50% | Conservative compression |
| 4:1 | 25% | Moderate compression |
| 8:1 | 12.5% | Aggressive compression |
| 16:1 | 6.25% | Extreme compression (optional) |

### Reference Results from Papers

| Paper | Model | Compression | PPL Degradation |
|-------|-------|-------------|-----------------|
| PALU (ICLR 2025) | Llama-2-7B | 50% | <0.1 |
| KVQuant (NeurIPS 2024) | Llama-2-7B | 3-bit quant | <0.1 |
| MiniCache (NeurIPS 2024) | Llama-2-7B | 5x | <0.5 |

### Evaluation Command

```bash
python scripts/benchmark/eval_perplexity.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset wikitext2 \
    --methods dense,lst,h2o,streaming,snapkv,pyramidkv,tova \
    --compression_ratios 2,4,8 \
    --seq_length 1024 \
    --num_samples 2048 \
    --output_file results/perplexity_llama2_7b.json
```

### Key Implementation Notes

1. **Stride**: Use stride = seq_length (non-overlapping windows)
2. **BOS token**: Include at start of each sequence
3. **Metric**: Report bits-per-byte (BPB) in addition to PPL for some papers
4. **GPU**: Single A100-80GB sufficient for 7B models

---

## 2. Zero-Shot Accuracy

**Priority: CRITICAL** — Standard reasoning preservation benchmark.

### Tasks (6 Standard)

| Task | Type | Metric | Description |
|------|------|--------|-------------|
| **PIQA** | Multiple choice | acc | Physical intuition QA |
| **WinoGrande** | Multiple choice | acc | Coreference resolution |
| **HellaSwag** | Multiple choice | acc_norm | Commonsense completion |
| **ARC-Easy** | Multiple choice | acc_norm | Science QA (easy) |
| **ARC-Challenge** | Multiple choice | acc_norm | Science QA (hard) |
| **OpenBookQA** | Multiple choice | acc_norm | Open-domain reasoning |

### Extended Tasks (Optional)

| Task | Type | Description |
|------|------|-------------|
| BoolQ | Boolean | Yes/No QA |
| COPA | Multiple choice | Causal reasoning |
| MathQA | Multiple choice | Mathematical reasoning |
| MMLU | Multiple choice | Multi-task benchmark (5-shot) |

### Evaluation Command

```bash
# Using lm-evaluation-harness (recommended for dense baseline)
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks piqa,winogrande,hellaswag,arc_easy,arc_challenge,openbookqa \
    --batch_size 8 \
    --num_fewshot 0 \
    --output_path results/zeroshot_dense.json

# Using our custom script (for compressed evaluation)
python scripts/benchmark/eval_zeroshot.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --methods dense,lst,h2o,streaming,snapkv,pyramidkv \
    --tasks piqa,winogrande,hellaswag,arc_easy,arc_challenge,openbookqa \
    --compression_ratio 2 \
    --output_file results/zeroshot_compressed.json
```

### Important Notes

1. **Short contexts**: Zero-shot tasks have 20-150 tokens, so compression impact is minimal at 2:1
2. **Higher ratios**: Test 4:1 and 8:1 to show where methods diverge
3. **Aggressive settings**: Use sink=4, recent=8 (instead of 32) to stress-test methods
4. **Few-shot**: Some papers report 5-shot accuracy (adds ~500 tokens of context)

### Reference Results (Llama-2-7B Dense)

| Task | Accuracy |
|------|----------|
| PIQA | 78.78% |
| WinoGrande | 69.61% |
| HellaSwag | 76.00% |
| ARC-Easy | 73.74% |
| ARC-Challenge | 44.97% |
| OpenBookQA | 44.20% |
| **Average** | **64.55%** |

---

## 3. LongBench

**Priority: CRITICAL** — Standard long-context understanding benchmark.

### Task Categories (17 Tasks Total)

#### Single-Document QA (3 tasks)
| Task | Avg Length | Description |
|------|------------|-------------|
| NarrativeQA | 18K | Story comprehension |
| Qasper | 5K | Scientific paper QA |
| MultiFieldQA-en | 5K | Multi-domain QA |

#### Multi-Document QA (3 tasks)
| Task | Avg Length | Description |
|------|------------|-------------|
| HotpotQA | 9K | Multi-hop reasoning |
| 2WikiMQA | 5K | Wikipedia multi-hop |
| MuSiQue | 11K | Complex multi-hop |

#### Summarization (3 tasks)
| Task | Avg Length | Description |
|------|------------|-------------|
| GovReport | 8K | Government report summarization |
| QMSum | 11K | Query-based meeting summarization |
| MultiNews | 2K | Multi-document news summarization |

#### Few-Shot Learning (3 tasks)
| Task | Avg Length | Description |
|------|------------|-------------|
| TREC | 5K | Question classification |
| TriviaQA | 8K | Trivia QA with context |
| SAMSum | 6K | Dialogue summarization |

#### Code (2 tasks)
| Task | Avg Length | Description |
|------|------------|-------------|
| LCC | 2K | Code completion |
| RepoBench-P | 5K | Repository-level code |

#### Synthetic (2 tasks)
| Task | Avg Length | Description |
|------|------------|-------------|
| PassageCount | 11K | Count passage occurrences |
| PassageRetrieval-en | 10K | Retrieve specific passage |

### KV Cache Sizes to Test

Following PyramidKV paper methodology:

| KV Size | Compression (vs 4K) | Compression (vs 8K) |
|---------|---------------------|---------------------|
| 64 | 64x | 128x |
| 128 | 32x | 64x |
| 256 | 16x | 32x |
| 512 | 8x | 16x |
| 1024 | 4x | 8x |
| 2048 | 2x | 4x |
| Full | 1x | 1x |

### Model Requirements

| Model | Context Window | Recommended |
|-------|----------------|-------------|
| Llama-2-7B | 4K | NO (truncation dominates) |
| Mistral-7B-Instruct-v0.2 | 32K | YES (primary) |
| Llama-3-8B-Instruct | 8K | YES (secondary) |
| Llama-3-70B-Instruct | 8K | Optional (quality check) |

### Evaluation Command

```bash
python scripts/benchmark/eval_longbench.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --methods dense,lst,h2o,snapkv,pyramidkv,streaming \
    --kv_cache_sizes 128,256,512,1024,2048 \
    --tasks narrativeqa,hotpotqa,qasper,gov_report,multifieldqa_en,2wikimqa \
    --output_dir results/longbench/
```

### Metrics

- **F1 Score**: Primary metric for QA tasks
- **ROUGE-L**: For summarization tasks
- **Exact Match**: For synthetic tasks
- **Pass@1**: For code tasks

---

## 4. Needle-in-a-Haystack (NIAH)

**Priority: HIGH** — Tests retrieval capability at various context lengths and depths.

### Test Setup

| Parameter | Values |
|-----------|--------|
| Context lengths | 1K, 4K, 8K, 16K, 32K |
| Depth percentages | 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100% |
| Needle | "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day." |
| Haystack | Paul Graham essays (or random text) |

### Depth Percentage Meaning

- **0%**: Needle at the very beginning
- **50%**: Needle in the middle
- **100%**: Needle at the very end

### Evaluation Command

```bash
python scripts/benchmark/eval_niah.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --methods dense,lst,h2o,snapkv,pyramidkv \
    --context_lengths 1024,4096,8192,16384,32768 \
    --depth_percents 0,10,20,30,40,50,60,70,80,90,100 \
    --num_trials 10 \
    --output_dir results/niah/
```

### Output Format

Generate a heatmap visualization where:
- X-axis: Context length
- Y-axis: Depth percentage
- Color: Retrieval accuracy (green=100%, red=0%)

### Key Findings from Literature

1. GPT-4 struggles with needles at document start as context increases
2. All models perform better when needle is near the end
3. Compression methods often fail at middle depths first

---

## 5. RULER Benchmark

**Priority: HIGH** — NVIDIA's comprehensive long-context benchmark.

### Task Categories (13 Tasks)

#### Retrieval (4 tasks)
| Task | Description |
|------|-------------|
| Single NIAH | Find one needle in haystack |
| Multi-key NIAH | Find needles matching multiple keys |
| Multi-value NIAH | Find multiple values for one key |
| Multi-query NIAH | Answer multiple queries about needles |

#### Multi-hop Tracing (2 tasks)
| Task | Description |
|------|-------------|
| Variable Tracking | Track variable assignments across context |
| Common Words | Find words appearing in multiple passages |

#### Aggregation (2 tasks)
| Task | Description |
|------|-------------|
| Common Words | Find most common words in context |
| Frequent Words | Count word frequencies |

#### Question Answering (5 tasks)
| Task | Description |
|------|-------------|
| Single QA | Answer question from single passage |
| Multi QA | Answer question requiring multiple passages |

### Context Lengths

| Length | Samples per Task |
|--------|------------------|
| 4K | 500 |
| 8K | 500 |
| 16K | 500 |
| 32K | 500 |
| 64K | 500 |
| 128K | 500 |

### Evaluation Command

```bash
python scripts/benchmark/eval_ruler.py \
    --model_name meta-llama/Llama-3-8B-Instruct \
    --methods dense,lst,h2o,snapkv,pyramidkv \
    --context_lengths 4096,8192,16384,32768 \
    --output_dir results/ruler/
```

### Reference: Llama-2-7B Baseline at 4K

Score: **85.6%** (used as threshold for "effective context length")

---

## 6. Efficiency Metrics

**Priority: HIGH** — System-level performance gains.

### Metrics to Report

| Metric | Unit | Description |
|--------|------|-------------|
| Decode Throughput | tokens/sec | Generation speed |
| Time to First Token (TTFT) | ms | Latency to start generating |
| KV Cache Memory | GB | Memory usage for KV cache |
| Peak GPU Memory | GB | Maximum GPU memory used |
| Prefill Latency | ms | Time to process prompt |

### Test Configurations

| Context Length | Batch Sizes |
|----------------|-------------|
| 2K | 1, 4, 8, 16 |
| 4K | 1, 4, 8, 16 |
| 8K | 1, 4, 8 |
| 16K | 1, 4 |
| 32K | 1 |

### Evaluation Command

```bash
python scripts/benchmark/eval_efficiency.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --methods dense,lst,h2o,streaming \
    --context_lengths 2048,4096,8192,16384 \
    --batch_sizes 1,4,8,16 \
    --num_warmup 5 \
    --num_trials 20 \
    --output_file results/efficiency.json
```

### Hardware Specification

Always report:
- GPU model (e.g., NVIDIA A100-80GB)
- CUDA version
- PyTorch version
- Number of GPUs
- Precision (FP16, BF16, FP32)

---

## 7. Model Selection Guide

### Recommended Models for Paper

| Model | Parameters | Context | Use For | Train Sidecar? |
|-------|------------|---------|---------|----------------|
| **Llama-2-7B** | 7B | 4K | Perplexity, Zero-shot | YES |
| **Llama-2-13B** | 13B | 4K | Scaling analysis | Optional |
| **Mistral-7B-Instruct-v0.2** | 7B | 32K | LongBench, NIAH, RULER | YES |
| **Llama-3-8B-Instruct** | 8B | 8K | LongBench, RULER | YES |
| **Llama-3-70B-Instruct** | 70B | 8K | Scaling analysis | Optional |

### Model Configurations

#### Llama-2-7B
```python
d_model = 4096
n_heads = 32
d_head = 128  # d_model / n_heads
n_layers = 32
vocab_size = 32000
```

#### Mistral-7B
```python
d_model = 4096
n_heads = 32
n_kv_heads = 8  # GQA
d_head = 128
n_layers = 32
vocab_size = 32000
sliding_window = 4096  # Note: affects KV cache behavior
```

#### Llama-3-8B
```python
d_model = 4096
n_heads = 32
n_kv_heads = 8  # GQA
d_head = 128
n_layers = 32
vocab_size = 128256
```

### GQA Considerations

Mistral-7B and Llama-3 use **Grouped Query Attention** (GQA) with 8 KV heads instead of 32:
- KV cache is 4x smaller per token
- Sidecar must handle n_kv_heads, not n_heads
- Compression ratio calculations differ

---

## 8. Baseline Implementations

### Methods to Compare Against

| Method | Type | Source | Notes |
|--------|------|--------|-------|
| H2O | Eviction | [GitHub](https://github.com/FMInference/H2O) | 20% heavy hitter budget |
| StreamingLLM | Window | [GitHub](https://github.com/mit-han-lab/streaming-llm) | sink=4, recent=window |
| SnapKV | Eviction | [Paper](https://arxiv.org/abs/2404.14469) | Observation window voting |
| PyramidKV | Eviction | [GitHub](https://github.com/Zefan-Cai/PyramidKV) | Layer-wise pyramid |
| TOVA | Eviction | [Paper](https://arxiv.org/abs/2401.06104) | Online eviction |
| Mean Pooling | Merging | Baseline | Simple average |

### Hyperparameters

#### H2O
```python
num_sink = 4
heavy_hitter_ratio = 0.2  # Keep top 20% by attention
```

#### StreamingLLM
```python
num_sink = 4
window_size = 1024  # Recent tokens to keep
```

#### SnapKV
```python
num_sink = 4
observation_window = 32  # Tokens to observe attention
pool_size = 7  # Pooling kernel size
```

#### PyramidKV
```python
num_sink = 4
num_recent = 32
pyramid_ratio = 2.0  # Lower layers get 2x more budget
```

---

## Quick Reference: What to Run

### Minimum Viable Evaluation

1. **Perplexity** on Llama-2-7B (WikiText-2) at 2:1, 4:1, 8:1
2. **Zero-Shot** on Llama-2-7B (6 tasks) at 2:1
3. **LongBench** on Mistral-7B (6 key tasks) at KV=512, 1024, 2048
4. **Efficiency** on Llama-2-7B at 4K, 8K context

### Full Evaluation

All of the above plus:
- Perplexity on C4
- Zero-shot at 4:1, 8:1 with aggressive settings
- Full LongBench (17 tasks)
- NIAH heatmaps
- RULER at multiple context lengths
- Llama-3-8B and Mistral-7B for all benchmarks

---

## References

- [MiniCache](https://arxiv.org/abs/2405.14366) - NeurIPS 2024
- [PALU](https://arxiv.org/abs/2407.21118) - ICLR 2025
- [KVQuant](https://arxiv.org/abs/2401.18079) - NeurIPS 2024
- [H2O](https://arxiv.org/abs/2306.14048) - NeurIPS 2023
- [PyramidKV](https://arxiv.org/abs/2406.02069) - 2024
- [SnapKV](https://arxiv.org/abs/2404.14469) - NeurIPS 2024
- [LongBench](https://github.com/THUDM/LongBench)
- [RULER](https://github.com/NVIDIA/RULER)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Needle-in-a-Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
