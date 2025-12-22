# LST: Learned Super-Token KV Cache Compression

*Query-Invariant Learned Compression for Long-Context LLMs*

**Target: ICML 2026**

---

## Abstract

We propose **LST (Learned Super-Token)**, a novel approach to KV cache compression that learns to compress windows of KV pairs into compact super-tokens. Unlike heuristic methods (H2O, StreamingLLM, TOVA) that prune tokens based on importance scores, or geometric methods (SnapKV, PyramidKV) that merge similar vectors, LST trains a lightweight sidecar network to produce optimal compressed representations.

**Key Innovation - Query-Probing Attention Alignment (QPAA):**
Standard perplexity training optimizes for specific continuations, but fails for novel queries. QPAA samples random probe queries during training to ensure super-tokens work for *any* future query, not just training distributions.

$$\mathcal{L}_{\text{QPAA}} = \mathbb{E}_{q \sim \mathcal{N}(0, I)} \left[ \left\| \text{Attn}(q, K_{\text{dense}}, V_{\text{dense}}) - \text{Attn}(q, K_{\text{comp}}, V_{\text{comp}}) \right\|_2^2 \right]$$

---

## Evaluation Benchmarks

Based on comprehensive analysis of accepted papers: [MiniCache](https://arxiv.org/abs/2405.14366) (NeurIPS 2024), [KVQuant](https://neurips.cc/virtual/2024/poster/96936) (NeurIPS 2024), [PALU](https://arxiv.org/abs/2407.21118) (ICLR 2025), [H2O](https://arxiv.org/abs/2306.14048) (NeurIPS 2023), [SnapKV](https://arxiv.org/abs/2404.14469), [PyramidKV](https://arxiv.org/abs/2406.02069).

| Benchmark | Priority | Purpose | Models | Source |
|-----------|----------|---------|--------|--------|
| **Perplexity** | CRITICAL | Primary quality metric (every paper) | Llama-2-7B/13B | WikiText-2, C4 |
| **Zero-Shot Accuracy** | CRITICAL | Reasoning preservation | Llama-2-7B | lm-evaluation-harness |
| **LongBench** | CRITICAL | Long-context understanding (17 tasks) | Mistral-7B-Instruct, Llama-3-8B | [LongBench](https://github.com/THUDM/LongBench) |
| **Needle-in-a-Haystack** | HIGH | Retrieval at depth/length | Any long-context | [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) |
| **RULER** | HIGH | Comprehensive long-context (13 tasks) | Any long-context | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) |
| **Efficiency** | HIGH | Throughput, latency, memory | Any | Custom profiling |

---

## Results

### Table 1: Perplexity (WikiText-2) — Primary Quality Metric

*Lower is better. Evaluated following [PALU](https://arxiv.org/abs/2407.21118) and [KVQuant](https://arxiv.org/abs/2306.14048) methodology.*

**Llama-2-7B at Multiple Compression Ratios:**

| Method | Type | 2:1 (50%) | 4:1 (25%) | 8:1 (12.5%) |
|--------|------|-----------|-----------|-------------|
| Dense (baseline) | — | 8.47 | 8.47 | 8.47 |
| **LST (Ours)** | learned | **9.46** | **13.56** | **15.97** |
| H2O | eviction | 15.81 | 28.37 | 31.80 |
| StreamingLLM | window | 25.08 | 25.08 | 25.08 |
| SnapKV | eviction | — | — | — |
| PyramidKV | eviction | — | — | — |
| TOVA | eviction | 15.87 | 28.68 | 31.86 |
| Mean Pooling | merging | 15.49 | 23.64 | 24.85 |
| CAM | merging | 15.98 | 28.9 | 32.31 |
| KVmerger | merging | 24.19 | 22.32 | 20.63 |
| Weightedkv | merging | 15.86 | 28.36 | 30.36 |

*Reference: PALU achieves <0.1 PPL degradation at 50% compression; KVQuant achieves <0.1 PPL degradation with 3-bit quantization.*

**Evaluation Settings:**
- Dataset: WikiText-2 test set (2048 random samples, seq_len=1024)
- Also report on C4 validation set for completeness
- Compression ratios defined as: original_tokens / compressed_tokens

---

### Table 2: Zero-Shot Accuracy (LM-Evaluation-Harness)

*Following [MiniCache](https://arxiv.org/abs/2405.14366) and [PALU](https://arxiv.org/abs/2407.21118) methodology: 6 standard tasks.*

**Llama-2-7B at 2:1 Compression (50% KV budget):**

| Method | PIQA | WinoGrande | HellaSwag | ARC-e | ARC-c | OBQA | Avg |
|--------|------|------------|-----------|-------|-------|------|-----|
| Dense | 78.78 | 69.61 | 76.00 | 73.74 | 44.97 | 44.20 | **64.55** |
| **LST (Ours)** | — | — | — | — | — | — | — |
| H2O | — | — | — | — | — | — | — |
| StreamingLLM | — | — | — | — | — | — | — |
| SnapKV | — | — | — | — | — | — | — |
| PyramidKV | — | — | — | — | — | — | — |

**Evaluation Settings:**
- Framework: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Tasks: `piqa,winogrande,hellaswag,arc_easy,arc_challenge,openbookqa`
- Zero-shot (num_fewshot=0)
- Report `acc_norm` for HellaSwag, ARC, OBQA; `acc` for others

**Note:** Zero-shot tasks have short contexts (20-150 tokens). Differences emerge at higher compression ratios (4:1, 8:1) or with aggressive settings (sink=4, recent=8).

---

### Table 3: LongBench (Long-Context Understanding)

*Following [PyramidKV](https://arxiv.org/abs/2406.02069) and [SnapKV](https://arxiv.org/abs/2404.14469) methodology. 17 tasks across 6 categories.*

**Mistral-7B-Instruct-v0.2 with KV Cache Size = 2048:**

| Method | Single-QA | Multi-QA | Summ | Few-shot | Code | Synth | Avg |
|--------|-----------|----------|------|----------|------|-------|-----|
| Full KV | — | — | — | — | — | — | — |
| **LST (Ours)** | — | — | — | — | — | — | — |
| H2O | — | — | — | — | — | — | — |
| SnapKV | — | — | — | — | — | — | — |
| PyramidKV | — | — | — | — | — | — | — |
| StreamingLLM | — | — | — | — | — | — | — |

**Task Categories (17 total):**
| Category | Tasks |
|----------|-------|
| Single-Doc QA | NarrativeQA, Qasper, MultiFieldQA-en |
| Multi-Doc QA | HotpotQA, 2WikiMQA, MuSiQue |
| Summarization | GovReport, QMSum, MultiNews |
| Few-shot | TREC, TriviaQA, SAMSum |
| Code | LCC, RepoBench-P |
| Synthetic | PassageCount, PassageRetrieval-en |

**Evaluation Settings:**
- Models: Mistral-7B-Instruct-v0.2 (32K context), Llama-3-8B-Instruct (8K context)
- KV cache sizes: 64, 128, 256, 512, 1024, 2048, Full
- Framework: [LongBench](https://github.com/THUDM/LongBench)

---

### Table 4: Needle-in-a-Haystack (NIAH)

*Retrieval accuracy at various context lengths and needle depths. Following [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) methodology.*

**Mistral-7B-Instruct-v0.2 at 2:1 Compression:**

| Method | 4K | 8K | 16K | 32K |
|--------|----|----|-----|-----|
| Full KV | 100% | 100% | 100% | 100% |
| **LST (Ours)** | — | — | — | — |
| H2O | — | — | — | — |
| SnapKV | — | — | — | — |
| PyramidKV | — | — | — | — |

**Evaluation Settings:**
- Context lengths: 1K, 4K, 8K, 16K, 32K
- Depth percentages: 0%, 25%, 50%, 75%, 100% (needle placement)
- Metric: Retrieval accuracy (exact match)
- Visualize as heatmap (context length × depth)

---

### Table 5: RULER (Comprehensive Long-Context)

*Following [NVIDIA RULER](https://github.com/NVIDIA/RULER) benchmark. 13 tasks across 4 categories.*

**Llama-3-8B-Instruct at Various Context Lengths:**

| Method | 4K | 8K | 16K | 32K |
|--------|----|----|-----|-----|
| Full KV | — | — | — | — |
| **LST (Ours)** | — | — | — | — |
| H2O | — | — | — | — |
| SnapKV | — | — | — | — |
| PyramidKV | — | — | — | — |

**Task Categories:**
- **Retrieval:** Single/Multi-key NIAH, Multi-value, Multi-query
- **Multi-hop Tracing:** Variable tracking across context
- **Aggregation:** Common/Frequent words extraction
- **Question Answering:** Long-context QA

**Evaluation Settings:**
- Context lengths: 4K, 8K, 16K, 32K, 64K, 128K
- 500 examples per task per length
- Inference: vLLM with BFloat16

---

### Table 6: Efficiency (Throughput & Memory)

*Measured on NVIDIA A100-80GB. Following [H2O](https://arxiv.org/abs/2306.14048) and [MiniCache](https://arxiv.org/abs/2405.14366) methodology.*

**Llama-2-7B Decode Throughput:**

| Method | Compression | 4K ctx | 8K ctx | 16K ctx | Memory (16K) |
|--------|-------------|--------|--------|---------|--------------|
| Dense | 1:1 | — tok/s | — tok/s | — tok/s | — GB |
| **LST (Ours)** | 2:1 | — tok/s | — tok/s | — tok/s | — GB |
| **LST (Ours)** | 4:1 | — tok/s | — tok/s | — tok/s | — GB |
| H2O | 2:1 | — tok/s | — tok/s | — tok/s | — GB |
| StreamingLLM | 2:1 | — tok/s | — tok/s | — tok/s | — GB |

**Metrics:**
- Decode throughput (tokens/second)
- Time to first token (TTFT)
- KV cache memory usage (GB)
- Batch size = 1, 4, 8, 16

---

## Baselines

| Method | Paper | Venue | Type | Approach |
|--------|-------|-------|------|----------|
| **H2O** | [Zhang et al.](https://arxiv.org/abs/2306.14048) | NeurIPS 2023 | Eviction | Cumulative attention scoring |
| **StreamingLLM** | [Xiao et al.](https://arxiv.org/abs/2309.17453) | ICLR 2024 | Window | Sink tokens + sliding window |
| **SnapKV** | [Li et al.](https://arxiv.org/abs/2404.14469) | NeurIPS 2024 | Eviction | Observation window voting |
| **PyramidKV** | [Cai et al.](https://arxiv.org/abs/2406.02069) | 2024 | Eviction | Layer-wise pyramidal budget |
| **TOVA** | [Oren et al.](https://arxiv.org/abs/2401.06104) | 2024 | Eviction | Online token eviction |
| **KVMerger** | [Wang et al.](https://arxiv.org/abs/2407.08454) | 2024 | Merging | Gaussian kernel weighted |
| **CaM** | [Zhang et al.](https://arxiv.org/abs/2405.14366) | ICML 2024 | Merging | Importance-weighted merging |
| **MiniCache** | [Liu et al.](https://arxiv.org/abs/2405.14366) | NeurIPS 2024 | Merging | Cross-layer KV merging |
| **PALU** | [Chi et al.](https://arxiv.org/abs/2407.21118) | ICLR 2025 | Low-rank | Low-rank projection |
| **KVQuant** | [Hooper et al.](https://arxiv.org/abs/2401.18079) | NeurIPS 2024 | Quantization | Per-channel KV quantization |

---

## Target Models

| Model | Context | Perplexity | Zero-Shot | LongBench | NIAH/RULER |
|-------|---------|------------|-----------|-----------|------------|
| **Llama-2-7B** | 4K | ✓ | ✓ | ✗ (truncation) | ✗ |
| **Llama-2-13B** | 4K | ✓ | ✓ | ✗ (truncation) | ✗ |
| **Mistral-7B-Instruct-v0.2** | 32K | ✓ | ✓ | ✓ | ✓ |
| **Llama-3-8B-Instruct** | 8K | ✓ | ✓ | ✓ | ✓ |

---

## Method

### Architecture

LST uses a lightweight **Sidecar Network** (~2M parameters) to compress windows of KV pairs:

```
Input: (batch, window_size, 2*d_head) — concatenated K and V

    ┌─────────────────────────────┐
    │ Input Projection            │  → Linear(2*d_head, hidden_dim)
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │ + Positional Embedding      │  → Learnable positions
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │ Transformer Encoder (2L)    │  → Cross-position dependencies
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │ Attention Pooling           │  → Learned query aggregation
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │ Output Projection           │  → Linear(hidden_dim, 2*d_head)
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │ Residual + Hard Norm        │  → Critical for PPL stability
    └─────────────────────────────┘

Output: (batch, 1, 2*d_head) — super-token (split into k̃, ṽ)
```

### Training Objective

$$\mathcal{L} = \lambda_{\text{ppl}} \mathcal{L}_{\text{PPL}} + \lambda_{\text{qpaa}} \mathcal{L}_{\text{QPAA}} + \lambda_{\text{div}} \mathcal{L}_{\text{Diversity}}$$

| Loss | Weight | Purpose |
|------|--------|---------|
| **PPL** | 1.0 | Language modeling quality |
| **QPAA** | 0.5 | Query-invariant attention (NOVEL) |
| **Diversity** | 0.1 | Prevent super-token collapse |

---

## Quick Start

### Training LST Sidecar

```bash
python scripts/train_sidecar.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir ./checkpoints/lst_llama2_2x \
    --window_size 2 \
    --max_steps 5000 \
    --batch_size 512
```

### Evaluating Perplexity

```bash
python scripts/benchmark/eval_perplexity.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --sidecar_path ./checkpoints/lst_llama2_2x/final.pt \
    --methods dense,lst,h2o,streaming,snapkv,pyramidkv \
    --compression_ratios 2,4,8
```

### Running Zero-Shot Evaluation

```bash
# Using lm-evaluation-harness
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks piqa,winogrande,hellaswag,arc_easy,arc_challenge,openbookqa \
    --batch_size 8 \
    --num_fewshot 0
```

### Running LongBench

```bash
python scripts/benchmark/eval_longbench.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --sidecar_path ./checkpoints/lst_mistral_2x/final.pt \
    --methods dense,lst,h2o,snapkv,pyramidkv \
    --kv_cache_sizes 128,256,512,1024,2048
```

### Running NIAH

```bash
python scripts/benchmark/eval_niah.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --sidecar_path ./checkpoints/lst_mistral_2x/final.pt \
    --context_lengths 4096,8192,16384,32768 \
    --depth_percents 0,25,50,75,100
```

---

## Project Structure

```
src/
├── LST/                    # Learned Super-Token module
│   ├── sidecar/           # Compression network
│   ├── losses/            # Training objectives (PPL, QPAA, Diversity)
│   └── training/          # Trainer and dataset
├── baselines/             # H2O, StreamingLLM, TOVA, SnapKV, PyramidKV, etc.
scripts/
├── train_sidecar.py       # Training script
└── benchmark/             # Evaluation scripts
    ├── eval_perplexity.py # WikiText-2, C4 perplexity
    ├── eval_zeroshot.py   # lm-eval-harness wrapper
    ├── eval_longbench.py  # 17-task long-context
    ├── eval_niah.py       # Needle-in-a-haystack
    ├── eval_ruler.py      # NVIDIA RULER benchmark
    └── eval_efficiency.py # Throughput, latency, memory
```

---

## References

**KV Cache Compression:**
- Zhang et al., "[H2O: Heavy-Hitter Oracle for Efficient Generative Inference](https://arxiv.org/abs/2306.14048)" (NeurIPS 2023)
- Xiao et al., "[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)" (ICLR 2024)
- Li et al., "[SnapKV: LLM Knows What You Are Looking For Before Generation](https://arxiv.org/abs/2404.14469)" (NeurIPS 2024)
- Cai et al., "[PyramidKV: Dynamic KV Cache Compression](https://arxiv.org/abs/2406.02069)" (2024)
- Liu et al., "[MiniCache: KV Cache Compression in Depth Dimension](https://arxiv.org/abs/2405.14366)" (NeurIPS 2024)
- Chi et al., "[PALU: KV-Cache Compression with Low-Rank Projection](https://arxiv.org/abs/2407.21118)" (ICLR 2025)
- Hooper et al., "[KVQuant: Towards 10M Context Length LLM Inference](https://arxiv.org/abs/2401.18079)" (NeurIPS 2024)

**Benchmarks:**
- [LongBench](https://github.com/THUDM/LongBench) - Long-context understanding benchmark
- [RULER](https://github.com/NVIDIA/RULER) - NVIDIA comprehensive long-context benchmark
- [Needle-in-a-Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) - Retrieval accuracy test
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Zero-shot evaluation framework

---
