# LST: Learned Super-Token KV Cache Compression

*Query-Invariant Learned Compression for Long-Context LLMs*

---

## Abstract

We propose **LST (Learned Super-Token)**, a novel approach to KV cache compression that learns to compress windows of KV pairs into compact super-tokens. Unlike heuristic methods (H2O, StreamingLLM, TOVA) that prune tokens based on importance scores, or geometric methods (ToMe, KVMerger) that merge similar vectors, LST trains a lightweight sidecar network to produce optimal compressed representations.

**Key Innovation - Query-Probing Attention Alignment (QPAA):**
Standard perplexity training optimizes for specific continuations, but fails for novel queries. QPAA samples random probe queries during training to ensure super-tokens work for *any* future query, not just training distributions.

$$\mathcal{L}_{\text{QPAA}} = \mathbb{E}_{q \sim \mathcal{N}(0, I)} \left[ \left\| \text{Attn}(q, K_{\text{dense}}, V_{\text{dense}}) - \text{Attn}(q, K_{\text{comp}}, V_{\text{comp}}) \right\|_2^2 \right]$$

---

## Baselines

We compare LST against 7 state-of-the-art KV cache compression methods:

| Method | Paper | Approach | Key Idea |
|--------|-------|----------|----------|
| **H2O** | Zhang et al., NeurIPS 2023 | Eviction | Keep tokens with highest cumulative attention scores |
| **StreamingLLM** | Xiao et al., ICLR 2024 | Window | Keep sink tokens + recent window only |
| **TOVA** | Oren et al., 2024 | Eviction | Token-wise greedy eviction based on attention |
| **ToMe** | Bolya et al., ICLR 2023 | Merging | Bipartite matching + average similar tokens |
| **KVMerger** | Wang et al., 2024 | Merging | Gaussian kernel weighted merging |
| **WeightedKV** | Yuan et al., ICASSP 2025 | Hybrid | Evict keys, merge values with attention weights |
| **CaM** | Zhang et al., 2024 | Hybrid | Cache merging with importance-weighted combination |

---

## Target Models

| Model | Parameters | Context | Architecture | Priority |
|-------|------------|---------|--------------|----------|
| **TinyLlama-1.1B** | 1.1B | 2K | LLaMA | Development |
| **Llama-2-7B** | 7B | 4K | LLaMA-2 | Primary |
| **Llama-3-8B** | 8B | 8K | LLaMA-3 | Primary |
| **Llama-3-70B** | 70B | 8K | LLaMA-3 | Scaling |
| **Llama-3.1-8B** | 8B | 128K | LLaMA-3.1 | Long-context |
| **Mistral-7B-v0.3** | 7B | 32K | Mistral | Alternative |

---

## Benchmarks

### Perplexity (WikiText-103)

**TinyLlama-1.1B - 8:1 Compression**

| Method | Compression | TinyLlama-1.1B | Llama-2-7B | Llama-3-8B | Llama-3-70B |
|--------|-------------|----------------|------------|------------|-------------|
| Dense (No Compression) | 1:1 | **11.22** | -- | -- | -- |
| **LST (Ours)** | 8:1 | 21.25 | -- | -- | -- |
| KVMerger | 8:1 | 24.13 | -- | -- | -- |
| Mean Pooling | 8:1 | 24.80 | -- | -- | -- |
| StreamingLLM | 8:1 | 25.50 | -- | -- | -- |
| WeightedKV | 8:1 | 26.42 | -- | -- | -- |
| H2O | 8:1 | 28.04 | -- | -- | -- |
| CaM | 8:1 | 28.19 | -- | -- | -- |
| TOVA | 8:1 | 28.22 | -- | -- | -- |

### Compression Ratio Ablation

**Llama-2-7b-hf (WikiText-103 PPL) - Baselines at Various Compression Ratios:**

| Method | Type | 2:1 | 3:1 | 4:1 | 5:1 | 6:1 | 7:1 | 8:1 |
|--------|------|-----|-----|-----|-----|-----|-----|-----|
| Dense | baseline | 8.80 | 8.80 | 8.80 | 8.80 | 8.80 | 8.80 | 8.80 |
| LST | merging | 8.95 | -- | -- | -- | -- | -- | -- |
| Mean Pooling | merging | 16.56 | 22.92 | 25.11 | 25.62 | 26.01 | 26.24 | 26.17 |
| StreamingLLM | eviction | 26.30 | 26.30 | 26.30 | 26.30 | 26.30 | 26.30 | 26.30 |
| WeightedKV | merging | 17.21 | 26.00 | 30.10 | 31.84 | 32.65 | 32.69 | 32.67 |
| H2O | eviction | 17.11 | 25.70 | 30.01 | 32.05 | 33.07 | 33.87 | 34.33 |
| KVMerger | merging | 43.16 | 33.83 | 29.48 | 26.87 | 25.32 | 24.57 | 24.07 |
| TOVA | eviction | 17.18 | 26.08 | 30.39 | 32.38 | 33.44 | 34.01 | 34.41 |
| CaM | merging | 17.27 | 26.27 | 30.65 | 32.73 | 33.87 | 34.40 | 34.83 |

**Key Observations:**
- **Mean Pooling** is surprisingly competitive at lower compression ratios (2:1 - 4:1)
- **KVMerger** struggles at aggressive compression (2:1) but improves at 6:1+
- **StreamingLLM** is compression-ratio independent (only keeps sink+recent tokens)
- **Eviction methods** (H2O, TOVA) show consistent ~26-29 PPL across ratios

### Needle-in-a-Haystack (NIAH)

Retrieval accuracy at different needle depths (0% = start, 100% = end):

**TinyLlama-1.1B (All Compression Ratios, context 256-512)**

| Method | 2:1 | 4:1 | 8:1 |
|--------|-----|-----|-----|
| Dense (No Compression) | **66.7%** | **66.7%** | **60.0%** |
| Mean Pooling | 0% | 0% | 0% |
| H2O | 0% | 0% | 0% |
| StreamingLLM | 0% | 0% | 0% |
| TOVA | 0% | 0% | 0% |

*Note: NIAH is extremely challenging for TinyLlama - **all compression methods fail at all ratios (2:1 to 8:1)**. Even dense only achieves ~60-67% due to model limitations at deeper needle positions. Larger models (Llama-2-7B+) are needed for meaningful NIAH evaluation.*

**Llama-2-7B (8:1 compression, 4K context)**

| Method | 0% | 25% | 50% | 75% | 100% | Avg |
|--------|-----|-----|-----|-----|------|-----|
| Dense | -- | -- | -- | -- | -- | -- |
| **LST (Ours)** | -- | -- | -- | -- | -- | -- |
| H2O | -- | -- | -- | -- | -- | -- |
| StreamingLLM | -- | -- | -- | -- | -- | -- |
| TOVA | -- | -- | -- | -- | -- | -- |

### LongBench

Multi-task long-context benchmark (F1 Score):

**TinyLlama-1.1B (Various Compression Ratios)**

| Method | 2:1 Avg | 4:1 Avg |
|--------|---------|---------|
| Dense | 2.3% | 2.3% |
| Mean Pooling | 0.7% | 0.5% |
| H2O | 0.8% | 1.1% |
| StreamingLLM | 0.2% | 0.2% |
| TOVA | 0.9% | 1.2% |

*Note: TinyLlama performs poorly on LongBench even without compression (~2% F1). Larger models (Llama-2-7B+) are required for meaningful benchmark scores.*

**Llama-2-7B (2:1 compression)**

| Method | NarrativeQA | Qasper | HotpotQA | GovReport | TriviaQA | Avg |
|--------|-------------|--------|----------|-----------|----------|-----|
| Dense | -- | -- | -- | -- | -- | -- |
| **LST (Ours)** | -- | -- | -- | -- | -- | -- |
| H2O | -- | -- | -- | -- | -- | -- |
| StreamingLLM | -- | -- | -- | -- | -- | -- |
| TOVA | -- | -- | -- | -- | -- | -- |

### Baseline Comparison Notes

**Important:** Baseline papers typically evaluate at less aggressive compression ratios than our default 8:1:

| Paper | Compression Tested | Cache Budget |
|-------|--------------------|--------------|
| H2O (NeurIPS 2023) | 5:1 | 20% |
| KVMerger (2024) | 2-3:1 | 35-50% |
| StreamingLLM (ICLR 2024) | Variable | Sink + Recent |
| TOVA (2024) | 3-5:1 | 20-30% |

Our 8:1 compression (12.5% budget) is significantly more aggressive than typical baseline evaluations.

---

## Method

### Architecture

LST uses a lightweight **Sidecar Network** (~1M parameters) to compress windows of KV pairs:

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

Multi-objective loss with warmup:

$$\mathcal{L} = \lambda_{\text{ppl}} \mathcal{L}_{\text{PPL}} + \lambda_{\text{qpaa}} \mathcal{L}_{\text{QPAA}} + \lambda_{\text{div}} \mathcal{L}_{\text{Diversity}}$$

| Loss | Weight | Purpose |
|------|--------|---------|
| **PPL** | 1.0 | Language modeling quality on compressed cache |
| **QPAA** | 0.5 | Query-invariant attention alignment (NOVEL) |
| **Diversity** | 0.1 | Prevent super-token collapse |

**QPAA (Query-Probing Attention Alignment):**
```python
def qpaa_loss(k_dense, v_dense, k_comp, v_comp, num_probes=8):
    """Ensure compressed cache works for ANY query."""
    loss = 0
    for _ in range(num_probes):
        q = torch.randn_like(k_dense[:, :, 0, :])  # Random probe query
        out_dense = attention(q, k_dense, v_dense)
        out_comp = attention(q, k_comp, v_comp)
        loss += F.mse_loss(out_comp, out_dense.detach())
    return loss / num_probes
```

### Cache Compression Strategy

During inference, LST preserves **sink tokens** and **recent tokens** while compressing the middle:

```
[SINK (4)] [COMPRESSED SUPER-TOKENS] [RECENT (8)]
    ↑              ↑                      ↑
  Always       Window → 1            Always
  preserved    (8:1 ratio)          preserved
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/anonymous/lst.git
cd lst

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

---

## Usage

### Training LST Sidecar

```bash
python scripts/train/train_lst.py \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output_dir ./checkpoints/lst \
    --max_steps 10000 \
    --batch_size 4 \
    --lambda_ppl 1.0 \
    --lambda_qpaa 0.5 \
    --lambda_diversity 0.1 \
    --num_probes 8 \
    --wandb_project lst_training
```

### Evaluating Perplexity

```bash
python scripts/benchmark/eval_perplexity.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --checkpoint ./checkpoints/lst/final.pt \
    --methods dense,lst,h2o,streaming,tova \
    --compression_ratio 8
```

### Running NIAH Benchmark

```bash
python scripts/benchmark/eval_niah.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --checkpoint ./checkpoints/lst/final.pt \
    --methods all \
    --context_lengths 2048,4096,8192 \
    --depths 0.0,0.25,0.5,0.75,1.0
```

### Running LongBench

```bash
python scripts/benchmark/eval_longbench.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --checkpoint ./checkpoints/lst/final.pt \
    --methods all \
    --tasks narrativeqa,qasper,hotpotqa
```

---

## Project Structure

```
src/
├── LST/                    # Learned Super-Token module
│   ├── sidecar/           # Compression network
│   │   └── network.py     # SidecarPPL architecture
│   ├── losses/            # Training objectives
│   │   ├── ppl.py         # Perplexity loss
│   │   ├── query_probing.py  # QPAA loss (NOVEL)
│   │   ├── diversity.py   # Collapse prevention
│   │   └── combined.py    # Multi-objective
│   ├── training/          # Training utilities
│   │   ├── trainer.py     # LSTTrainer class
│   │   └── dataset.py     # TextDataset
│   └── config.py          # Configuration
├── baselines/             # Comparison methods
│   ├── h2o.py            # H2O eviction
│   ├── streaming.py      # StreamingLLM
│   ├── tova.py           # TOVA eviction
│   ├── tome.py           # Token Merging
│   ├── kvmerger.py       # KVMerger
│   ├── weightedkv.py     # WeightedKV
│   └── cam.py            # CaM
scripts/
├── train/                 # Training scripts
│   └── train_lst.py
└── benchmark/             # Evaluation scripts
    ├── eval_perplexity.py
    ├── eval_niah.py
    └── eval_longbench.py
tests/
└── test_lst.py           # Unit tests
```

---

## References

- Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient KV Cache" (NeurIPS 2023)
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)
- Bolya et al., "Token Merging: Your ViT But Faster" (ICLR 2023)
- Oren et al., "Transformers are Multi-State RNNs" (2024)
- Wang et al., "Model Tells You Where to Merge: KVMerger" (2024)
- Yuan et al., "WeightedKV: Attention Scores Weighted KV Merging" (ICASSP 2025)
- Zhang et al., "CaM: Cache Merging for Memory-Efficient LLMs" (2024)
- Lee et al., "Set Transformer" (ICML 2019)

---

