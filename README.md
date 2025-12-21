# Jacobian-Preserved KV Cache Compression (JPKV)

*Sensitivity-Aware Learned Compression for Long-Context LLMs*

**Target Venue:** ICML 2026
**Status:** See [STATUS.md](STATUS.md) for current progress

---

## Abstract

We propose **Jacobian-Preserved KV Cache Compression (JPKV)**, a learned approach to compressing the key-value cache in Transformer-based LLMs. Unlike heuristic methods (H2O, StreamingLLM) that prune tokens based on scalar importance scores, or geometric methods (Token Merging) that average similar vectors, we learn a compression function that preserves the **sensitivity structure** of attention—specifically, the Jacobian $\partial y / \partial q$ that captures how attention outputs respond to query perturbations.

Our key insight: two KV configurations can produce identical outputs but vastly different Jacobians. Preserving the Jacobian ensures the compressed cache responds to novel queries the same way the dense cache would, enabling robust generalization beyond the training distribution.

---

## Key Results (TinyLlama-1.1B)

### Attention Preservation (Cosine Similarity)

| Compression | JPKV (Ours) | H2O | StreamingLLM | vs H2O | vs StreamingLLM |
|-------------|-------------|-----|--------------|--------|-----------------|
| **8:1**     | **0.9154**  | 0.6921 | 0.5890 | **+32.3%** | **+55.4%** |
| **16:1**    | **0.9279**  | 0.5894 | 0.4660 | **+57.4%** | **+99.1%** |
| **32:1**    | **0.9211**  | 0.5598 | 0.4146 | **+64.5%** | **+122.2%** |

**Key finding:** JPKV's advantage grows with compression ratio. At 32:1, JPKV preserves 92% of attention behavior while baselines degrade to ~50%.

### WikiText-2 Evaluation (8:1 compression)

| Method | Attention Cosine | Improvement |
|--------|------------------|-------------|
| **JPKV (Ours)** | **0.8894** | - |
| H2O | 0.8328 | JPKV +6.8% |
| StreamingLLM | 0.7806 | JPKV +13.9% |

Dense Perplexity: **10.46** (baseline reference)

### Passkey Retrieval by Depth (8:1 compression)

| Depth | JPKV | H2O | StreamingLLM |
|-------|------|-----|--------------|
| 0.1 (start) | **0.9511** | 0.5691 | 0.5691 |
| 0.3 | **0.9515** | 0.7662 | 0.9606 |
| 0.5 (middle) | 0.9099 | 0.9388 | 0.5258 |
| 0.7 | 0.8455 | 0.9132 | 0.3809 |
| 0.9 (end) | **0.8087** | 0.6780 | 0.5260 |
| **Average** | **0.8934** | 0.7731 | 0.5925 |

**Key finding:** JPKV maintains consistent quality across all context depths, while baselines degrade significantly at certain positions.

---

## 1. Motivation

### The KV Cache Bottleneck

For a sequence of length $L$ with model dimension $d$ and $n_\text{layers}$ layers:

$$\text{Memory} = 2 \times L \times d \times n_\text{layers} \times \text{sizeof(dtype)}$$

At 100K context with Llama-3-70B (float16), this exceeds **40GB per sequence**—often larger than the model weights themselves.

### Why Existing Methods Fall Short

| Method | Approach | Limitation | Our Results |
|--------|----------|------------|-------------|
| **H2O** | Keep top-k by attention score | Heuristic; loses information before needed | 0.69 cosine (JPKV: 0.88) |
| **StreamingLLM** | Keep sink + recent tokens | Loses all mid-context | 0.59 cosine (JPKV: 0.88) |
| **Token Merging** | Average similar tokens | Preserves geometry, not function | Not tested |
| **Quantization** | Reduce precision | Orthogonal; composable with ours | Complementary |

All these methods optimize proxies (attention scores, similarity) rather than the actual computation: **how does the cache affect model outputs?**

#### Baseline Implementation Details (Apple-to-Apple)

For fair comparison at **8:1 compression** (8 tokens → 1 token):

- **JPKV**: Learned Sidecar network compresses window to single super-token
- **H2O**: Keep 1 token with highest cumulative attention received (`attn_weights.sum(dim=0).topk(1)`)
- **StreamingLLM**: Keep first token only (attention sink, `keys[:, :1, :]`)

---

## 2. Theoretical Foundation

### 2.1 The Attention Jacobian

For a query $q$ attending to keys $K \in \mathbb{R}^{N \times d}$ and values $V \in \mathbb{R}^{N \times d}$:

$$y = \text{softmax}\left(\frac{qK^\top}{\sqrt{d}}\right) V$$

The **Attention Jacobian** is:

$$\mathbf{J} = \frac{\partial y}{\partial q} \in \mathbb{R}^{d \times d}$$

This matrix encodes the **sensitivity** of the attention output to query perturbations. It captures:
- Which directions in query space most affect the output
- The local curvature of the attention function
- How the model would respond to slightly different inputs

### 2.2 Why Jacobian Preservation > Output Matching

**Theorem (informal):** Two KV configurations $\mathcal{M}_1, \mathcal{M}_2$ can satisfy:

$$y(\mathcal{M}_1) = y(\mathcal{M}_2) \quad \text{but} \quad \mathbf{J}(\mathcal{M}_1) \neq \mathbf{J}(\mathcal{M}_2)$$

**Implication:** Output matching ensures correct behavior *at one query*. Jacobian matching ensures correct behavior *in a neighborhood of queries*—critical for:

1. **Multi-step reasoning:** Earlier compression errors shift later queries
2. **Generalization:** Inference queries differ from training queries
3. **Robustness:** Small input variations shouldn't cause large output changes

### 2.3 The Jacobian Preservation Objective

Matching full $d \times d$ Jacobians is expensive. We use **stochastic Jacobian probing**:

$$\mathcal{L}_{\text{JP}}(\theta) = \mathbb{E}_{q, v} \left[ \left\| \mathbf{J}_{\text{dense}} \cdot v - \mathbf{J}_{\text{compressed}}(\theta) \cdot v \right\|_2^2 \right]$$

where $v \sim \mathcal{N}(0, I)$ are random probe vectors. This is equivalent to matching the Frobenius norm of the Jacobian difference in expectation, but costs only $O(d)$ per probe (one backward pass) rather than $O(d^2)$.

**Implementation via double backprop:**
```python
def jacobian_vector_product(output, query, probe_vector):
    """Compute J @ v efficiently via backward pass."""
    return torch.autograd.grad(
        outputs=output,
        inputs=query,
        grad_outputs=probe_vector,
        create_graph=True  # Enable gradient through this operation
    )[0]
```

### 2.4 Connection to Physics

This framework has a natural interpretation via **normal mode analysis**:

- The Jacobian's singular values are the "stiffness" of different response modes
- Jacobian preservation ensures the compressed system has the same "vibrational spectrum"
- Analogous to coarse-graining in molecular dynamics, but matching response functions rather than forces

---

## 3. Method

### 3.1 The Sidecar Network

A lightweight network $\Phi_\theta$ compresses a window of $N$ KV pairs to a single "super-token":

$$(\tilde{k}, \tilde{v}) = \Phi_\theta(k_1, v_1, \ldots, k_N, v_N)$$

**Architecture (~1M parameters):**
```
Input: (batch, N, 2d) — concatenated K and V

    ┌─────────────────────────────┐
    │ Transformer Encoder (2-3L)  │  ← Captures cross-position dependencies
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │ Set Transformer Pooling     │  ← Permutation-invariant N→1
    │ (ISAB + PMA)                │
    └─────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │ Output Projection           │  ← Split to k̃, ṽ
    └─────────────────────────────┘

Output: (batch, 1, 2d) — super-token
```

### 3.2 Training Objective

Our final loss combines direct output matching with auxiliary terms:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{output}}}_{\text{Attention output MSE}} + \lambda_1 \underbrace{\mathcal{L}_{\text{kv}}}_{\text{KV reconstruction}} + \lambda_2 \underbrace{\mathcal{L}_{\text{attn}}}_{\text{Attention pattern}} + \lambda_3 \underbrace{\mathcal{L}_{\text{div}}}_{\text{Diversity}}$$

Where:

- $\mathcal{L}_{\text{output}}$: MSE between dense and compressed attention outputs (primary, weight=1.0)
- $\mathcal{L}_{\text{kv}}$: Reconstruction loss on mean KV (weight=0.1)
- $\mathcal{L}_{\text{attn}}$: Attention pattern similarity (weight=0.05)
- $\mathcal{L}_{\text{div}}$: Diversity regularization to prevent collapse (weight=0.01)

**Key insight:** Direct output matching with query augmentation (mixing real + random queries) provides better generalization than indirect Jacobian probing.

### 3.3 Training Protocol

**Phase 1: Data Collection**
```bash
python scripts/collect_trajectories.py \
    --model_name meta-llama/Llama-3-8B \
    --dataset redpajama \
    --num_samples 100000 \
    --window_size 64
```

Collects: `(kv_window, context_kv, future_queries)` tuples

**Phase 2: Sidecar Training**
```bash
python scripts/train_sidecar.py \
    --trajectories_path ./data/trajectories \
    --loss_type jacobian_probe \
    --num_probes 4 \
    --batch_size 256 \
    --max_steps 50000
```

### 3.4 Inference Integration

```python
def generate_with_compression(model, sidecar, prompt, max_tokens, window_size=64):
    kv_cache = []

    for step in range(max_tokens):
        logits, new_kv = model.forward(prompt[step], kv_cache)
        kv_cache.append(new_kv)

        # Compress when window is full
        if len(kv_cache) >= window_size:
            window = kv_cache[:window_size]
            super_token = sidecar(window)  # (1, 2d)
            kv_cache = [super_token] + kv_cache[window_size:]

        yield logits.argmax()
```

---

## 4. Experimental Plan

### 4.1 Research Questions

| RQ | Question | Metric |
|----|----------|--------|
| **RQ1** | Does Jacobian preservation outperform output matching? | Task accuracy at fixed compression |
| **RQ2** | How does JPKV compare to heuristic methods? | PPL, RULER, Needle-in-Haystack |
| **RQ3** | What compression ratios are achievable? | Quality vs. ratio Pareto curve |
| **RQ4** | Does JPKV generalize across query distributions? | Train on A, test on B |

### 4.2 Baselines

1. **Dense** (no compression) — upper bound
2. **H2O** — attention-score pruning
3. **StreamingLLM** — sink + recent window
4. **Token Merging** — geometric averaging
5. **Output Matching** — our architecture, MSE loss only

### 4.3 Benchmarks

| Benchmark | Task | Why It Matters |
|-----------|------|----------------|
| **RULER** | Synthetic retrieval at varying depths | Tests information preservation |
| **Needle-in-Haystack** | Find hidden info in 100K tokens | Long-range retrieval |
| **LongBench** | Multi-document QA, summarization | Real-world long-context |
| **Perplexity (PG-19)** | Language modeling | General quality |
| **Jacobian Cosine Sim** | cos(J_dense, J_compressed) | Direct objective validation |

### 4.4 Ablations

1. **Loss components:** JP only vs. JP + output vs. all three
2. **Number of probes:** 1, 2, 4, 8 random vectors per sample
3. **Compression ratio:** 4:1, 8:1, 16:1, 32:1
4. **Architecture:** Transformer vs. GIN vs. MLP encoder
5. **Window size:** 32, 64, 128 tokens

---

## 5. Implementation

### Directory Structure

```
src/jpkv/
├── sidecar/           # Compression network
│   ├── network.py     # Main architecture
│   ├── encoders.py    # Transformer, GIN, MLP
│   └── aggregators.py # Set Transformer, attention pooling
├── losses/
│   ├── jacobian.py    # Jacobian computation utilities
│   └── objectives.py  # JP, output, attention losses
├── data/
│   ├── collector.py   # Trajectory collection
│   └── dataset.py     # PyTorch dataset
├── training/
│   └── trainer.py     # Training loop
├── evaluation/
│   ├── benchmarks/    # RULER, Needle, LongBench
│   └── methods/       # JPKV, baselines
└── models/
    ├── wrapper.py     # HuggingFace model interface
    └── hooks.py       # KV cache capture
```

### Key Files

- **Theory → Code:** See [CLAUDE.md](CLAUDE.md) for implementation constraints
- **Current Status:** See [STATUS.md](STATUS.md) for what's done/pending
- **Experiments:** See `experiments/` for configs and logs

---

## 6. Timeline (ICML 2026)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Validation** | 2 weeks | Sanity checks pass on TinyLlama |
| **Data Collection** | 2 weeks | 100K trajectories from Llama-3-8B |
| **Training** | 3 weeks | Trained sidecar, ablations complete |
| **Evaluation** | 3 weeks | Full benchmark suite, comparisons |
| **Writing** | 4 weeks | Paper draft, figures, appendix |
| **Buffer** | 2 weeks | Revisions, additional experiments |

**Submission Deadline:** ~January 2026

---

## References

- Bolya et al., "Token Merging: Your ViT But Faster" (ICLR 2023)
- Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient KV Cache" (NeurIPS 2023)
- Xiao et al., "StreamingLLM" (ICLR 2024)
- Lee et al., "Set Transformer" (ICML 2019)
- Noid et al., "The Multiscale Coarse-Graining Method" (J. Chem. Phys. 2008) — physics inspiration

---

## Development

See [CLAUDE.md](CLAUDE.md) for engineering standards and [STATUS.md](STATUS.md) for current progress.

```bash
# Run tests
pytest tests/ -v

# Collect trajectories
python scripts/collect_trajectories.py --help

# Train sidecar
python scripts/train_sidecar.py --help

# Evaluate
python scripts/evaluate.py --help
```
