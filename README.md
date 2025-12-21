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
| ToMe | 8:1 | -- | -- | -- | -- |

### Needle-in-a-Haystack (NIAH)

Retrieval accuracy at different needle depths (0% = start, 100% = end):

**TinyLlama-1.1B (8:1 compression, 4K context)**

| Method | 0% | 25% | 50% | 75% | 100% | Avg |
|--------|-----|-----|-----|-----|------|-----|
| Dense | -- | -- | -- | -- | -- | -- |
| **LST (Ours)** | -- | -- | -- | -- | -- | -- |
| H2O | -- | -- | -- | -- | -- | -- |
| StreamingLLM | -- | -- | -- | -- | -- | -- |
| TOVA | -- | -- | -- | -- | -- | -- |
| ToMe | -- | -- | -- | -- | -- | -- |
| KVMerger | -- | -- | -- | -- | -- | -- |
| WeightedKV | -- | -- | -- | -- | -- | -- |
| CaM | -- | -- | -- | -- | -- | -- |

**Llama-2-7B (8:1 compression, 4K context)**

| Method | 0% | 25% | 50% | 75% | 100% | Avg |
|--------|-----|-----|-----|-----|------|-----|
| Dense | -- | -- | -- | -- | -- | -- |
| **LST (Ours)** | -- | -- | -- | -- | -- | -- |
| H2O | -- | -- | -- | -- | -- | -- |
| StreamingLLM | -- | -- | -- | -- | -- | -- |
| TOVA | -- | -- | -- | -- | -- | -- |
| ToMe | -- | -- | -- | -- | -- | -- |
| KVMerger | -- | -- | -- | -- | -- | -- |
| WeightedKV | -- | -- | -- | -- | -- | -- |
| CaM | -- | -- | -- | -- | -- | -- |

**Llama-3-8B (8:1 compression, 8K context)**

| Method | 0% | 25% | 50% | 75% | 100% | Avg |
|--------|-----|-----|-----|-----|------|-----|
| Dense | -- | -- | -- | -- | -- | -- |
| **LST (Ours)** | -- | -- | -- | -- | -- | -- |
| H2O | -- | -- | -- | -- | -- | -- |
| StreamingLLM | -- | -- | -- | -- | -- | -- |
| TOVA | -- | -- | -- | -- | -- | -- |
| ToMe | -- | -- | -- | -- | -- | -- |
| KVMerger | -- | -- | -- | -- | -- | -- |
| WeightedKV | -- | -- | -- | -- | -- | -- |
| CaM | -- | -- | -- | -- | -- | -- |

### LongBench

Multi-task long-context benchmark (F1/ROUGE-L/Accuracy):

**Llama-2-7B (8:1 compression)**

| Method | NarrativeQA | Qasper | HotpotQA | GovReport | TriviaQA | Avg |
|--------|-------------|--------|----------|-----------|----------|-----|
| Dense | -- | -- | -- | -- | -- | -- |
| **LST (Ours)** | -- | -- | -- | -- | -- | -- |
| H2O | -- | -- | -- | -- | -- | -- |
| StreamingLLM | -- | -- | -- | -- | -- | -- |
| TOVA | -- | -- | -- | -- | -- | -- |
| ToMe | -- | -- | -- | -- | -- | -- |
| KVMerger | -- | -- | -- | -- | -- | -- |
| WeightedKV | -- | -- | -- | -- | -- | -- |
| CaM | -- | -- | -- | -- | -- | -- |

**Llama-3-8B (8:1 compression)**

| Method | NarrativeQA | Qasper | HotpotQA | GovReport | TriviaQA | Avg |
|--------|-------------|--------|----------|-----------|----------|-----|
| Dense | -- | -- | -- | -- | -- | -- |
| **LST (Ours)** | -- | -- | -- | -- | -- | -- |
| H2O | -- | -- | -- | -- | -- | -- |
| StreamingLLM | -- | -- | -- | -- | -- | -- |
| TOVA | -- | -- | -- | -- | -- | -- |
| ToMe | -- | -- | -- | -- | -- | -- |
| KVMerger | -- | -- | -- | -- | -- | -- |
| WeightedKV | -- | -- | -- | -- | -- | -- |
| CaM | -- | -- | -- | -- | -- | -- |

### Compression Ratio Ablation

Quality vs. compression trade-off (Llama-2-7B, WikiText-2 PPL):

| Method | 4:1 | 8:1 | 16:1 | 32:1 |
|--------|-----|-----|------|------|
| **LST (Ours)** | -- | -- | -- | -- |
| H2O | -- | -- | -- | -- |
| StreamingLLM | -- | -- | -- | -- |
| TOVA | -- | -- | -- | -- |
| ToMe | -- | -- | -- | -- |
| KVMerger | -- | -- | -- | -- |
| WeightedKV | -- | -- | -- | -- |
| CaM | -- | -- | -- | -- |

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

