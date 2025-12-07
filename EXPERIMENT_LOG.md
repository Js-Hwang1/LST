# FMKVC Experiment Log

## Experiment v1: Force-Weighted Consistency Training

**Date:** 2024-12-06
**Run ID:** fmkv_force_matching/59v8cbrx
**Hardware:** Google Colab A100 40GB

---

## 1. Trajectory Collection

### 1.1 Source Model
- **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Architecture:** 32 heads, d_head = 64, hidden_dim = 2048
- **Layers collected:** All 22 layers

### 1.2 Data Statistics
- **Total trajectory files:** 49
- **Windows per file:** ~3200
- **Total windows:** ~159,500
- **Total data size:** ~50 GB

### 1.3 Trajectory Window Structure
Each window contains:
```
keys:          (window_size=64, d_head=64)     # K states
values:        (window_size=64, d_head=64)     # V states
queries:       (num_queries=16, d_head=64)     # Future Q states
forces:        (window_size=64, hidden_dim=2048)  # dL/dK, dL/dV
force_queries: (num_queries=16, hidden_dim=2048)  # dL/dQ (TRUE forces)
```

### 1.4 Force Collection Method
Forces were computed via backpropagation through the frozen LLM:
```
F_kv(t) = nabla_{h_t} L_LM
```
where L_LM is the language modeling loss at future positions.

---

## 2. Training Configuration

### 2.1 Sidecar Architecture
- **Encoder:** 3-layer Transformer (d=256, 4 heads, ffn_ratio=4)
- **Aggregator:** Set Transformer with 8 inducing points
- **Parameters:** 4,374,144
- **Input:** Concatenated K+V: (batch, window_size, 2*d_head)
- **Output:** Compressed super-token: (batch, 1, 2*d_head)

### 2.2 Training Hyperparameters
```
learning_rate:      1e-4
warmup_steps:       1000
max_steps:          5000
batch_size:         64
num_windows:        4 (per sample)
consistency_weight: 0.1
force_weight:       1.0
geometric_weight:   0.01
```

### 2.3 Multi-Window Setup
For non-trivial Jacobians, we use 4 consecutive windows:
```
Dense sequence:  [W_1, W_2, W_3, W_4] -> 256 tokens
Compressed:      [c_1, c_2, c_3, c_4] -> 4 super-tokens
Compression:     64x
```

---

## 3. Loss Function

### 3.1 Total Loss
```
L_total = w_c * L_consistency + w_f * L_force + w_g * L_geometric
```

### 3.2 Consistency Loss (Force-Weighted)
```
L_consistency = sum_i [ ||y_dense(q_i) - y_cg(q_i)||^2 * w_i ]

where:
  y_dense = softmax(Q @ K_dense.T / sqrt(d)) @ V_dense
  y_cg    = softmax(Q @ K_cg.T / sqrt(d)) @ V_cg
  w_i     = ||F_query(q_i)|| / sum_j ||F_query(q_j)||  # Force-based importance
```

### 3.3 Force Matching Loss (Disabled in v1)
```
L_force = || J_dense - J_cg ||_F^2

where:
  J = d(y)/d(q)  # Attention Jacobian
```

### 3.4 Geometric Losses
- Anti-collapse: Penalizes low-rank outputs
- Diversity: Penalizes cosine similarity between windows

---

## 4. Results

### 4.1 Training Metrics
```
Initial loss:  580,000,000
Final loss:    11.94
Steps:         5000
Time:          ~8 minutes
```

### 4.2 WandB Final Metrics
```
eval/loss:                    12.145
train/jacobian/cg_norm:       0.143
train/jacobian/dense_norm:    4.937
train/jacobian/ratio:         0.020  # CRITICAL: Only 2%!
train/force/using_force_weights: 1
```

### 4.3 Checkpoint Analysis

#### Scale Analysis
| Metric | Dense | Compressed | Ratio |
|--------|-------|------------|-------|
| K norm (per token) | 6.996 | 0.010 | 0.001x |
| V norm (per token) | 0.086 | 0.027 | 0.31x |
| Attention entropy | 5.545 | 1.386 | 0.25x |
| Attention max | 0.004 | 0.250 | 62.5x |

#### Jacobian Analysis
| Metric | Dense | Compressed | Ratio |
|--------|-------|------------|-------|
| Jacobian norm | 0.329 | 0.000 | ~0.00003x |
| Per-query norm | 0.011 | 0.000 | ~0.00003x |

#### Output Quality
| Metric | Value |
|--------|-------|
| Output MSE | 0.000010 |
| Output cosine similarity | 0.7525 |

---

## 5. Diagnosis

### 5.1 Primary Issue: Jacobian Collapse
The compressed attention Jacobians are 36,000x smaller than dense attention.

**Root Cause Analysis:**

1. **Scale Collapse**
   - k_cg norm = 0.01 vs k_dense norm = 7.0 (700x smaller)
   - The sidecar learns to shrink K values toward zero

2. **Softmax Saturation**
   - CG attention entropy = 1.39 (very peaky, near one-hot)
   - Dense attention entropy = 5.55 (distributed)
   - When softmax saturates, gradients vanish: d(softmax)/d(x) -> 0

3. **Output Match Deception**
   - Low MSE (0.00001) and decent cosine sim (0.75)
   - The model found a "shortcut": make attention peaky, but still produce similar outputs
   - This kills the Jacobians while keeping outputs superficially close

### 5.2 Mathematical Explanation

For scaled dot-product attention:
```
a_i = exp(q @ k_i / sqrt(d)) / sum_j exp(q @ k_j / sqrt(d))
```

When ||k_cg|| << ||k_dense||:
- Attention scores become similar: q @ k_cg ~ 0 for all k
- Softmax produces uniform distribution initially
- But with only 4 tokens, softmax saturates to 0.25 each
- The Jacobian d(a)/d(q) = a_i * (I - a) @ k approaches zero

---

## 6. Next Steps

### 6.1 Proposed Fixes

**Fix 1: Output Normalization**
```
k_cg_normalized = k_cg * (||k_dense||_mean / ||k_cg||)
v_cg_normalized = v_cg * (||v_dense||_mean / ||v_cg||)
```

**Fix 2: Direct Jacobian Loss**
```
L_jacobian = || J_dense - J_cg ||_F^2

J = d(y)/d(q) = A @ V @ d(A)/d(q)
  = A @ V @ (diag(a) - a @ a.T) @ K.T / sqrt(d)
```

**Fix 3: Scale-Invariant Consistency**
```
L_consistency = 1 - cosine_similarity(y_dense, y_cg)
```
This prevents the model from collapsing scale while matching directions.

**Fix 4: Stronger Anti-Collapse**
Increase anti-collapse weight or add explicit norm matching:
```
L_norm = (||k_cg|| - ||k_dense||)^2 + (||v_cg|| - ||v_dense||)^2
```

### 6.2 Priority Order
1. Add norm matching loss (simplest)
2. Switch to cosine similarity consistency
3. Add explicit Jacobian matching loss
4. Tune temperature for softmax (learned scaling)

---

## 7. Key Insights

1. **Force-weighted consistency alone is insufficient.** The model can minimize weighted MSE without preserving gradients.

2. **Scale freedom is problematic.** Without explicit constraints, the sidecar collapses to near-zero magnitudes.

3. **Jacobian preservation requires explicit supervision.** Matching outputs does not guarantee matching derivatives.

4. **The 64x compression is aggressive.** May need intermediate compression ratios (16x, 32x) first.

---

## Appendix: Code References

- Trajectory collection: `scripts/collect_trajectories.py`
- Training: `scripts/train_sidecar.py`
- Loss function: `src/fmkv/losses/force_matching.py`
- Analysis: `scripts/analyze_checkpoint.py`
- v1 WandB: https://wandb.ai/hwang30916-stony-brook-university/fmkv_force_matching/runs/59v8cbrx

---

# Experiment v2: Re-enabled Physics with Manifold Regularization

**Date:** 2024-12-06
**Run ID:** fmkv_v2_physics/d9prv19e
**Hardware:** Google Colab A100 40GB
**WandB:** https://wandb.ai/hwang30916-stony-brook-university/fmkv_v2_physics/runs/d9prv19e

---

## 1. Changes from v1

### 1.1 Problem Diagnosis from v1

The reasoning agent identified that **physics was disabled**:
- `force_matching_weight = 0.0` meant Jacobian matching was completely turned off
- Only consistency (output MSE) was being minimized
- This allowed the sidecar to find a "shortcut" solution: collapse k_cg to zero

### 1.2 Loss Function Changes

**v1 Loss Weights (Disabled Physics):**
```
force_matching_weight:    0.0   # DISABLED - no Jacobian matching!
consistency_weight:       50.0  # Dominated
output_magnitude_weight:  5.0
kv_match_weight:         10.0
```

**v2 Loss Weights (Re-enabled Physics):**
```
force_matching_weight:    10.0  # RE-ENABLED - primary objective
consistency_weight:       1.0   # Reduced - base requirement
output_magnitude_weight:  1.0
kv_match_weight:         1.0   # Reduced - direction only
kv_norm_weight:          0.1   # NEW - manifold regularization
```

### 1.3 New Loss: KV Norm Matching (Manifold Regularization)

The key fix is forcing compressed tokens to inhabit the same latent manifold as dense tokens:

```
L_norm = (||k_cg|| - mu_dense^K)^2 + (||v_cg|| - mu_dense^V)^2

where:
  mu_dense^K = (1/N) * sum_i ||k_i||   # Mean norm of dense keys
  mu_dense^V = (1/N) * sum_i ||v_i||   # Mean norm of dense values
```

This prevents the scale collapse ||k_cg|| -> 0 that destroyed Jacobians in v1.

---

## 2. Mathematical Justification

### 2.1 Why Scale Collapse Kills Jacobians

For attention with compressed keys:
```
a = softmax(q @ k_cg^T / sqrt(d))
```

When ||k_cg|| -> 0:
1. q @ k_cg -> 0 for all queries
2. softmax input approaches zero vector
3. softmax output approaches uniform distribution u = [1/N, ..., 1/N]
4. Jacobian d(softmax)/d(q) = diag(a) - a @ a^T -> 0 (isotropic, small)

### 2.2 The Corrected Objective

```
L_total = lambda_1 * L_consistency + lambda_2 * L_force + lambda_3 * L_norm

where:
  L_consistency = ||y_dense - y_cg||^2           # Output matching
  L_force = ||J_dense - J_cg||_F^2 / ||J_dense||_F^2  # Relative Jacobian matching
  L_norm = (||k_cg|| - mu_dense^K)^2 + (||v_cg|| - mu_dense^V)^2  # Scale matching
```

**Hyperparameters:**
- lambda_1 = 1.0 (base requirement)
- lambda_2 = 10.0 (primary objective - steering)
- lambda_3 = 0.1 (soft constraint for stability)

---

## 3. v2 Training Results

### 3.1 Training Metrics
```
Initial loss:  ~580,000,000 (same starting point as v1)
Final loss:    5089.2
Steps:         5000
Time:          ~9.5 minutes
```

### 3.2 WandB Final Metrics
```
eval/loss:                    2063.68
train/jacobian/cg_norm:       49.75
train/jacobian/dense_norm:    4.97
train/jacobian/ratio:         24.25   # Was 0.02 in v1 - 1200x improvement!
train/kv_norm/k_cg:           30.625  # Was 0.01 in v1 - no longer collapsed!
train/kv_norm/k_dense:        19.125
train/kv_norm/v_cg:           10.0625
train/kv_norm/v_dense:        1.55
train/force/using_force_weights: 1
```

### 3.3 v1 vs v2 Comparison

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| jacobian/cg_norm | 0.143 | 49.75 | +348x |
| jacobian/dense_norm | 4.937 | 4.97 | ~same |
| jacobian/ratio | 0.020 | 24.25 | +1212x |
| kv_norm/k_cg | 0.010 | 30.625 | +3062x |
| kv_norm/k_dense | 6.996 | 19.125 | +2.7x |
| kv_norm/v_cg | ~0.027 | 10.0625 | +373x |
| kv_norm/v_dense | 0.086 | 1.55 | +18x |
| eval/loss | 12.145 | 2063.68 | +170x |

---

## 4. v2 Analysis

### 4.1 Primary Achievement: Jacobian Collapse Fixed

The critical issue from v1 - Jacobian collapse - has been resolved:
- v1: ||J_cg|| / ||J_dense|| = 0.02 (Jacobian was 50x too small)
- v2: ||J_cg|| / ||J_dense|| = 24.25 (Jacobian is now 24x too large)

The force matching loss with weight=10.0 successfully prevents collapse by penalizing
the relative Jacobian difference:
```
L_force = ||J_dense - J_cg||_F^2 / ||J_dense||_F^2
```

### 4.2 New Issue: Scale Explosion

While v1 suffered from **scale collapse**, v2 exhibits the opposite problem - **scale explosion**:

| Component | Dense | Compressed | Ratio |
|-----------|-------|------------|-------|
| K norm | 19.125 | 30.625 | 1.60x (CG > Dense!) |
| V norm | 1.55 | 10.06 | 6.49x (CG >> Dense!) |
| Jacobian | 4.97 | 49.75 | 10.0x (CG >> Dense!) |

The sidecar is producing super-tokens with **larger** norms than the original tokens.

### 4.3 Mathematical Explanation of Scale Explosion

The Jacobian of scaled dot-product attention w.r.t. query q is:
```
J = d(y)/d(q) = A @ V @ (diag(a) - a @ a^T) @ K^T / sqrt(d)
```

where a = softmax(q @ K^T / sqrt(d)).

For the Jacobian norm, we have approximately:
```
||J|| ~ ||K|| * ||V|| * f(entropy(a))
```

In v1, ||K_cg|| -> 0 caused ||J_cg|| -> 0 (collapse).
In v2, ||K_cg|| = 30.6 > ||K_dense|| = 19.1, so ||J_cg|| >> ||J_dense|| (explosion).

The kv_norm_weight = 0.1 was too weak to constrain the scale.

### 4.4 Why the Force Matching Loss Drives Scale Up

The force matching loss in v2 is:
```
L_force = ||J_dense - J_cg||_F^2 / ||J_dense||_F^2
```

When ||J_cg|| < ||J_dense||, the gradient w.r.t. K_cg is:
```
d(L_force)/d(K_cg) ~ (J_cg - J_dense) * ... -> pushes K_cg UP
```

The loss is minimized when J_cg = J_dense, but with only 4 super-tokens vs 256 dense tokens,
the sidecar must compensate with larger magnitudes to produce similar Jacobian norms.

The problem: with 4 tokens, the softmax entropy is inherently lower (max ~1.39 bits vs ~5.5 bits).
Lower entropy means the Jacobian factor (diag(a) - a @ a^T) is smaller.
To compensate, the sidecar increases ||K|| and ||V|| to boost ||J||.

---

## 5. Diagnosis and Proposed Fixes

### 5.1 Root Cause

The force matching loss prioritizes Jacobian matching (lambda=10.0) over norm matching (lambda=0.1).
With only 4 super-tokens, the sidecar cannot match Jacobian norms without scale explosion
because the softmax entropy is fundamentally lower.

### 5.2 Mathematical Constraint

For N_cg=4 super-tokens vs N_dense=256 dense tokens, the maximum entropy differs:
```
H_max(4) = log(4) = 1.386 bits
H_max(256) = log(256) = 5.545 bits
```

This entropy gap directly affects the Jacobian:
```
||J||^2 ~ ||K||^2 * ||V||^2 * Var(a)
Var(a) ~ 1/N when uniform, smaller when peaked
```

To match ||J_dense|| with lower entropy, we need ||K_cg|| * ||V_cg|| >> ||K_dense|| * ||V_dense||.

### 5.3 Proposed Fixes for v3

**Fix 1: Bidirectional Norm Matching**

Replace one-sided norm matching with bidirectional constraint:
```
L_norm = max(||k_cg|| - mu_K, 0)^2 + max(mu_K - ||k_cg||, 0)^2
       = (||k_cg|| - mu_K)^2  # Already symmetric, but increase weight
```
Increase kv_norm_weight: 0.1 -> 1.0 to make it competitive with force_matching.

**Fix 2: Reduce Force Matching Weight**

The force matching weight is too aggressive:
```
force_matching_weight: 10.0 -> 1.0
kv_norm_weight:        0.1 -> 1.0
```
This balances Jacobian matching against scale preservation.

**Fix 3: Relative Force Matching with Scale Penalty**

Add explicit scale penalty to force matching:
```
L_force = ||J_dense - J_cg||_F^2 / ||J_dense||_F^2
        + beta * (||J_cg||_F / ||J_dense||_F - 1)^2

where beta = 1.0 encourages ||J_cg|| = ||J_dense||
```

**Fix 4: Temperature Scaling**

Instead of letting the sidecar learn arbitrary scales, enforce:
```
K_cg_scaled = K_cg * (||K_dense||_mean / ||K_cg||_mean)
V_cg_scaled = V_cg * (||V_dense||_mean / ||V_cg||_mean)
```
This normalizes the super-tokens before attention, preventing both collapse and explosion.

### 5.4 Priority Order for v3

1. **Increase kv_norm_weight** from 0.1 to 1.0 (simplest)
2. **Decrease force_matching_weight** from 10.0 to 1.0 (rebalance)
3. Add explicit **Jacobian ratio penalty** to prevent explosion
4. Consider **explicit temperature/scale normalization** layer

---

## 6. Key Insights from v2

1. **Force matching works** - re-enabling the Jacobian loss with weight=10.0 completely eliminated the collapse from v1.

2. **Scale regularization was too weak** - kv_norm_weight=0.1 could not compete with force_matching_weight=10.0.

3. **Entropy mismatch is fundamental** - with 4 tokens vs 256, the sidecar must learn larger-than-natural scales to match Jacobian behavior.

4. **The loss balance is critical** - v1 had force=0.0 (collapse), v2 has force=10.0, norm=0.1 (explosion). v3 needs force ~ norm for equilibrium.

5. **Compression ratio affects scale** - 64x compression (64 tokens -> 1 super-token) may require explicit scale matching or a different loss formulation.

---

## Appendix: v2 Code Changes

### Loss Function Modifications (src/fmkv/losses/force_matching.py)

```python
# v2 Fix: Re-enabled force matching
force_matching_weight: float = 10.0  # Was 0.0 in v1

# v2 Fix: Added KV norm matching
kv_norm_weight: float = 0.1  # NEW

# v2 Fix: KV Norm Matching Loss computation
k_dense_norm_mean = keys.norm(dim=-1).mean(dim=-1)
v_dense_norm_mean = values.norm(dim=-1).mean(dim=-1)
k_cg_norm_mean = k_cg.norm(dim=-1).mean(dim=-1)
v_cg_norm_mean = v_cg.norm(dim=-1).mean(dim=-1)

kv_norm_loss = (k_cg_norm_mean - k_dense_norm_mean).pow(2) + \
               (v_cg_norm_mean - v_dense_norm_mean).pow(2)
```

### Key Metric Additions
- `kv_norm/k_cg`, `kv_norm/k_dense`: Track K scale ratio
- `kv_norm/v_cg`, `kv_norm/v_dense`: Track V scale ratio
- `loss/kv_norm`: Track norm matching loss component

---

# Experiment v3: Cosine-Magnitude Decomposition (FAILED)

**Date:** 2024-12-06
**Run ID:** fmkv_v3_manifold/qy4lx4uy
**Hardware:** Google Colab A100 40GB
**WandB:** https://wandb.ai/hwang30916-stony-brook-university/fmkv_v3_manifold/runs/qy4lx4uy

---

## 1. Changes from v2

### 1.1 v3 Loss Function Design

Based on reasoning agent's recommendation, v3 decomposed force matching into:

**Cosine Direction Matching:**
```
L_dir = 1 - cos(J_dense, J_cg)
```
Goal: Match the steering direction independent of magnitude.

**Log Magnitude Matching:**
```
L_mag = |log(||J_dense||) - log(||J_cg||)|
```
Goal: Match intensity on log scale to handle wide dynamic range.

**Ratio-based Manifold Regularization:**
```
L_manifold = (||K_cg||/||K_dense|| - 1)^2 + (||V_cg||/||V_dense|| - 1)^2
```
Goal: Enforce K_cg and V_cg norms match dense norms exactly.

### 1.2 v3 Loss Weights
```python
consistency_weight:      1.0
force_direction_weight:  5.0   # L_dir = 1 - cos(J_dense, J_cg)
force_magnitude_weight:  1.0   # L_mag = |log norm diff|
manifold_weight:         2.0   # Ratio-based
# Legacy (disabled)
force_matching_weight:   0.0
kv_norm_weight:          0.0
```

### 1.3 Hyperparameters
- batch_size: 256
- learning_rate: 5e-5
- max_steps: 2200
- num_windows_per_sample: 4

---

## 2. v3 Training Results

### 2.1 Final Metrics
```
train/loss/total:            64000
eval/loss:                   33322.67
train/jacobian/ratio:        105.5    # WORSE than v2's 24.25!
train/jacobian/cg_norm:      326      # EXPLODED (v2 was 49.75)
train/jacobian/dense_norm:   5.81
train/jacobian/cosine_sim:   -0.0016  # ORTHOGONAL! (should be ~1.0)
train/kv_norm/k_cg:          54.5     # v2 was 30.6
train/kv_norm/k_dense:       19.125
train/kv_norm/v_cg:          30.625   # v2 was 10.06
train/kv_norm/v_dense:       1.88
train/manifold/k_ratio:      3.42     # Should be 1.0
train/manifold/v_ratio:      38.75    # Should be 1.0 - MASSIVELY OFF!
train/loss/force_direction:  1.0      # cos=0, completely orthogonal
```

### 2.2 v1 vs v2 vs v3 Comparison

| Metric | v1 | v2 | v3 | Trend |
|--------|----|----|-----|-------|
| jacobian/ratio | 0.02 | 24.25 | 105.5 | 📈 WORSE |
| jacobian/cg_norm | 0.14 | 49.75 | 326 | 📈 WORSE |
| jacobian/cosine_sim | N/A | N/A | -0.0016 | ❌ ORTHOGONAL |
| kv_norm/k_cg | 0.01 | 30.6 | 54.5 | 📈 WORSE |
| kv_norm/v_cg | 0.03 | 10.06 | 30.6 | 📈 WORSE |
| manifold/k_ratio | N/A | N/A | 3.42 | ❌ FAILED |
| manifold/v_ratio | N/A | N/A | 38.75 | ❌ FAILED |
| eval/loss | 12.1 | 2063.7 | 33322.7 | 📈 WORSE |

---

## 3. v3 Failure Analysis

### 3.1 Critical Finding: Jacobians are Orthogonal

The most alarming result is `cosine_sim = -0.0016`:
- The compressed attention Jacobian is **completely orthogonal** to the dense Jacobian
- The sidecar is producing gradients that steer in a **random direction**
- This is worse than random - it's essentially noise

### 3.2 Why Cosine-Magnitude Decomposition Failed

**Problem 1: Gradient Conflict**

The cosine direction loss `L_dir = 1 - cos(J_dense, J_cg)` has gradient:
```
d(L_dir)/d(J_cg) = -J_dense / (||J_dense|| * ||J_cg||) + cos * J_cg / ||J_cg||^2
```

This pushes J_cg toward J_dense's direction. But the magnitude loss pushes J_cg's norm toward ||J_dense||.

When K_cg and V_cg are scaled up (by anti-collapse or other losses), the Jacobian magnitude increases but the direction drifts because the attention pattern changes with scale.

**Problem 2: Manifold Weight Too Weak**

The manifold_weight=2.0 was dominated by:
- `anti_collapse`: 41728
- `kv_match`: 7584
- `manifold`: 7136

The anti-collapse penalty (which prevents K_cg from being small) is 6x larger than manifold regularization!

**Problem 3: Log Magnitude is Unstable**

`L_mag = |log(||J_dense||) - log(||J_cg||)|` has issues:
- When ||J_cg|| >> ||J_dense||, the gradient is small (1/||J_cg||)
- This provides weak corrective signal when the scale is already exploded

### 3.3 Why Ratio-Based Manifold Failed

The manifold loss `(||K_cg||/||K_dense|| - 1)^2` has:
- `k_ratio = 54.5 / 19.125 = 2.85` (logged as 3.42?)
- `v_ratio = 30.6 / 1.88 = 16.3` (logged as 38.75?)

The loss is `(2.85-1)^2 + (16.3-1)^2 = 3.4 + 234 = 237.4`, but logged as 7136.

The weight=2.0 gives 2 * 237 = 474, but this is tiny compared to anti_collapse=41728.

---

## 4. Root Cause Diagnosis

### 4.1 The Anti-Collapse Penalty is Too Strong

Looking at the loss breakdown:
```
anti_collapse:     41728  (65% of total)
kv_match:          7584   (12%)
manifold:          7136   (11%)
consistency:       11.6   (0.02%)
force_direction:   1.0    (cosine loss, not used effectively)
force_magnitude:   4.2    (tiny)
```

The anti-collapse penalty dominates everything. It's designed to prevent K_cg from collapsing to zero, but it's pushing K_cg to be **much larger** than necessary.

### 4.2 Direction Matching Has No Teeth

`force_direction_weight = 5.0` gives `5 * 1.0 = 5.0` contribution.
This is 0.008% of the total loss - completely ignored!

The direction loss of 1.0 means `cos(J_dense, J_cg) = 0` - completely orthogonal.
But the optimizer doesn't care because other losses are 10000x larger.

### 4.3 The Fundamental Problem: Loss Scale Mismatch

The consistency loss uses MSE between attention outputs:
- When norms are ~50, MSE can be ~2500
- When norms explode to 326, the Jacobian squared errors are ~100,000

The force matching losses were designed when norms were reasonable, but now they're dominated by the explosion dynamics.

---

## 5. Proposed Fixes for v4

### 5.1 Fix 1: Disable Anti-Collapse or Rebalance

The anti-collapse penalty was added to prevent v1's collapse. But v2 and v3 show the opposite problem - explosion.

**Option A:** Disable anti-collapse entirely
```python
anti_collapse_weight = 0.0
```

**Option B:** Make it symmetric (penalize both collapse AND explosion)
```python
L_anti = (||K_cg|| - ||K_dense||).abs().pow(2)  # Symmetric
```

### 5.2 Fix 2: Hard Normalization in Architecture

Instead of soft regularization, enforce scale matching architecturally:
```python
def compress_cache(self, keys, values):
    k_cg, v_cg = self.encoder(keys, values)

    # Hard normalize to match input norms
    k_cg = k_cg * (keys.norm(dim=-1, keepdim=True).mean() / k_cg.norm(dim=-1, keepdim=True))
    v_cg = v_cg * (values.norm(dim=-1, keepdim=True).mean() / v_cg.norm(dim=-1, keepdim=True))

    return k_cg, v_cg
```

This guarantees `||K_cg|| = ||K_dense||` by construction.

### 5.3 Fix 3: Normalize Jacobians Before Comparison

```python
J_dense_normed = J_dense / (||J_dense|| + eps)
J_cg_normed = J_cg / (||J_cg|| + eps)
L_direction = ||J_dense_normed - J_cg_normed||^2
```

This focuses purely on direction matching without scale interference.

### 5.4 Fix 4: Gradient Penalty on K/V Norms

Instead of loss-based regularization, clip gradients that would increase K/V norms:
```python
if k_cg.norm() > k_dense.norm() * 1.5:
    k_cg.register_hook(lambda grad: grad * 0.1)  # Dampen growth
```

### 5.5 Priority Order for v4

1. **Hard normalization** - Architectural fix that guarantees norm matching
2. **Disable anti-collapse** - Remove the main driver of explosion
3. **Normalize Jacobians** - Compare directions only
4. **Rebalance loss weights** - Make manifold competitive with other losses

---

## 6. Key Insights from v3

1. **Cosine-magnitude decomposition doesn't work** when other losses dominate. The direction loss contributed 0.008% of total loss.

2. **Anti-collapse penalty is the enemy** - It's 65% of total loss and drives explosion.

3. **Soft regularization fails** when loss scales are mismatched by 10000x.

4. **Architectural constraints beat loss functions** - Hard normalization would have prevented both collapse and explosion.

5. **The optimizer follows the gradient** - If manifold loss is 2000x smaller than anti-collapse, it's ignored.

6. **v1→v2→v3 progression: collapse→explosion→worse explosion** - We're oscillating around the optimum without finding it.

---

## 7. Summary

| Version | Problem | Jacobian Ratio | K_cg Norm | V_cg Norm |
|---------|---------|----------------|-----------|-----------|
| v1 | Collapse | 0.02 | 0.01 | 0.03 |
| v2 | Explosion | 24.25 | 30.6 | 10.06 |
| v3 | Worse Explosion + Orthogonal | 105.5 | 54.5 | 30.6 |
| v4 (target) | Balanced | ~1.0 | ~19.1 | ~1.9 |

**Next step:** Implement hard normalization in the Sidecar architecture to guarantee norm matching, then disable anti-collapse penalty.
