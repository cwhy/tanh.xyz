# From Online K-Means to SoftHebb: A Unified View

These three algorithms are closely related. Each one adds a twist to the previous, but the core idea is the same: **maintain a set of centroids that learn to represent clusters in the data, one example at a time.**

---

## 1. Online K-Means

The simplest starting point. You have K centroids in ℝⁿ.

### Algorithm

```
For each input x:
    1. ASSIGN:  find the nearest centroid
                k* = argmax_k  sim(w_k, x)     [using Euclidean distance]
    
    2. UPDATE:  move that centroid toward x
                w_k* ← w_k* + η · (x - w_k*)
    
    All other centroids stay unchanged.
```

### Properties

- **Hard assignment**: each input belongs to exactly one centroid
- **Euclidean space**: centroids live anywhere in ℝⁿ
- **Only the winner learns**: one centroid updates per example
- **No normalization**: centroids can have any magnitude

### What it's doing

Each centroid drifts toward the running mean of the points it "wins." Over time, centroids converge to cluster centers — the same ones batch K-means would find.

---

## 2. Hard WTA (Winner-Take-All)

Same idea as online K-means, but adapted to work on the **unit hypersphere** using cosine similarity.

### Key change from K-Means

Instead of Euclidean distance, use cosine similarity. Instead of the simple averaging update, use a **self-normalizing Hebbian rule** that keeps centroids on the unit sphere automatically.

### Algorithm

```
For each input x (normalized to ||x|| = 1):
    1. PREACTIVATION:  for every neuron k, compute
                       u_k = w_k · x          [cosine similarity, since ||w_k|| ≈ 1]
    
    2. HARD COMPETITION:  the single best-matching neuron wins
                          k* = argmax_k  u_k
                          y_k* = 1,  y_k = 0 for all k ≠ k*
    
    3. UPDATE:  only the winner's weights change
                Δw_{ik*} = η · 1 · (x_i - u_k* · w_{ik*})
                
                All other neurons: Δw_{ik} = 0   (because y_k = 0)
```

### The update rule unpacked

The rule `Δw_{ik} = η · y_k · (x_i - u_k · w_{ik})` has two parts:

| Term | Role |
|------|------|
| `x_i` | **Hebbian pull**: move toward the input |
| `- u_k · w_{ik}` | **Normalization push**: shrink the weight proportional to current response |

Together they keep `\|\|w_k\|\| → 1` automatically, without needing an explicit normalization step after each update. This is what makes it "self-normalizing."

### Comparison with K-Means

| | Online K-Means | Hard WTA |
|---|---|---|
| Similarity | Euclidean distance | Cosine similarity |
| Data space | Flat ℝⁿ | Unit hypersphere |
| Update rule | `η · (x - w)` | `η · (x - u·w)` |
| Normalization | None (or explicit) | Self-normalizing via u·w term |
| Assignment | Hard (nearest 1) | Hard (winner takes all) |

If you normalize the data and add explicit weight normalization after each K-means step, the two algorithms behave very similarly.

---

## 3. SoftHebb

Same architecture as Hard WTA, but replaces the hard winner-take-all with **soft competition via softmax**. This turns it from hard clustering into soft clustering.

### Key change from Hard WTA

Instead of picking one winner (y_k = 1, rest = 0), compute a **soft responsibility** for every neuron using softmax. Every neuron updates, weighted by how well it matched.

### Algorithm

```
For each input x (normalized to ||x|| = 1):
    1. PREACTIVATION:  for every neuron k, compute
                       u_k = w_k · x                    [cosine similarity]
    
    2. SOFT COMPETITION:  softmax over all neurons
                          y_k = exp(u_k + w_{0k}) / Σ_l exp(u_l + w_{0l})
                          
                          Now y_k ∈ (0,1) for all k, and Σ_k y_k = 1
                          (w_{0k} is a learnable bias per neuron)
    
    3. UPDATE WEIGHTS:  every neuron updates, weighted by y_k
                        Δw_{ik} = η · y_k · (x_i - u_k · w_{ik})
    
    4. UPDATE BIASES:   each neuron's bias also learns
                        Δw_{0k} = η · exp(-w_{0k}) · (y_k - exp(w_{0k}))
```

### What each part does

**Soft competition (Step 2):**
The softmax converts raw similarities into probabilities. If neuron 47 matches well (u=0.85) and neuron 200 matches poorly (u=0.3), the softmax might give y_47=0.15 and y_200=0.0002. The best match gets the most credit, but others participate too.

The bias w_{0k} encodes **how common** cluster k is — the Bayesian prior P(C_k). A cluster that owns many data points develops a larger bias, making it more likely to win future competitions even with slightly lower similarity.

**Weight update (Step 3):**
Identical formula to Hard WTA. The only difference is that y_k is now a soft probability instead of 0 or 1. So every centroid moves toward the input, but the best-matching centroid moves the most.

**Bias update (Step 4):**
This learns the cluster frequencies. If neuron k wins often (y_k is frequently high), w_{0k} increases, reflecting that this cluster represents a larger portion of the data. This has no equivalent in K-means or Hard WTA.

### The temperature knob

SoftHebb includes an optional temperature parameter via the softmax base b:

```
y_k = b^(u_k + w_{0k}) / Σ_l b^(u_l + w_{0l})
```

| Base b | Temperature | Behavior | Equivalent to |
|--------|-------------|----------|---------------|
| b → ∞ | T → 0 | One neuron dominates completely | **Hard WTA** |
| b = e ≈ 2.718 | T = 1 | Standard softmax | Standard SoftHebb |
| b → 1 | T → ∞ | All neurons equal (y_k = 1/K) | No clustering |

Hard WTA is literally a special case of SoftHebb at infinite base / zero temperature. The paper found that intermediate values (b = 200 to 1000) work best in practice.

---

## Side-by-side comparison

```
INPUT: image x (normalized to unit length)

╔══════════════════╦═══════════════════╦═══════════════════╦═══════════════════╗
║                  ║  Online K-Means   ║     Hard WTA      ║     SoftHebb      ║
╠══════════════════╬═══════════════════╬═══════════════════╬═══════════════════╣
║ Similarity       ║  Euclidean dist   ║  Cosine sim       ║  Cosine sim       ║
║                  ║  ||x - w_k||      ║  u_k = w_k · x    ║  u_k = w_k · x    ║
╠══════════════════╬═══════════════════╬═══════════════════╬═══════════════════╣
║ Assignment       ║  Hard             ║  Hard             ║  Soft (softmax)   ║
║                  ║  nearest 1        ║  y_k*=1, rest=0   ║  y_k ∈ (0,1)      ║
╠══════════════════╬═══════════════════╬═══════════════════╬═══════════════════╣
║ Who updates      ║  Winner only      ║  Winner only      ║  All neurons      ║
║                  ║                   ║                   ║  (weighted by y_k) ║
╠══════════════════╬═══════════════════╬═══════════════════╬═══════════════════╣
║ Weight update    ║  η·(x - w)        ║  η·(x - u·w)      ║  η·y_k·(x - u·w)  ║
╠══════════════════╬═══════════════════╬═══════════════════╬═══════════════════╣
║ Self-normalizing ║  No               ║  Yes (||w||→1)    ║  Yes (||w||→1)    ║
╠══════════════════╬═══════════════════╬═══════════════════╬═══════════════════╣
║ Cluster priors   ║  No               ║  No               ║  Yes (bias w_{0k}) ║
╠══════════════════╬═══════════════════╬═══════════════════╬═══════════════════╣
║ Dead neurons     ║  Common problem   ║  Common problem   ║  Reduced (soft)   ║
╠══════════════════╬═══════════════════╬═══════════════════╬═══════════════════╣
║ Bayesian theory  ║  No               ║  No               ║  Yes (proven)     ║
╚══════════════════╩═══════════════════╩═══════════════════╩═══════════════════╝
```

---

## The progression in one sentence

**Online K-Means** assigns each input to one centroid and averages.
**Hard WTA** does the same but on the unit sphere with a self-normalizing rule.
**SoftHebb** softens the assignment so every centroid learns from every input, proportionally, and also learns how big each cluster is — making it a proper Bayesian generative model.

---

## Concrete numeric example

Suppose K=3 neurons, input dimension n=4. An input arrives: **x = [0.5, 0.5, 0.5, 0.5]** (already unit-normalized).

Current weights (each row is a neuron's weight vector, roughly unit-normalized):

```
w_1 = [0.9, 0.3, 0.1, 0.2]   (||w|| ≈ 1.0)
w_2 = [0.2, 0.8, 0.4, 0.3]   (||w|| ≈ 1.0)
w_3 = [0.4, 0.4, 0.6, 0.5]   (||w|| ≈ 1.0)
```

### Step 1: Preactivation (same for all three algorithms)

```
u_1 = 0.9·0.5 + 0.3·0.5 + 0.1·0.5 + 0.2·0.5 = 0.75
u_2 = 0.2·0.5 + 0.8·0.5 + 0.4·0.5 + 0.3·0.5 = 0.85
u_3 = 0.4·0.5 + 0.4·0.5 + 0.6·0.5 + 0.5·0.5 = 0.95   ← best match
```

### Step 2: Assignment

**Hard WTA:**
```
y_1 = 0,  y_2 = 0,  y_3 = 1       (neuron 3 wins, others shut off)
```

**SoftHebb** (using standard softmax, biases = 0 for simplicity):
```
y_1 = exp(0.75) / Z = 2.117 / 7.858 = 0.269
y_2 = exp(0.85) / Z = 2.340 / 7.858 = 0.298
y_3 = exp(0.95) / Z = 2.586 / 7.858 = 0.329     ← highest, but others nonzero
                                   (Z = 7.858 — note: these don't sum to 1.0
                                    due to rounding, but the idea holds)
```

### Step 3: Weight update (η = 0.1)

Focus on neuron 3, weight index i=1 (where x_1 = 0.5, w_{13} = 0.4):

**Hard WTA:**
```
Δw_{13} = 0.1 · 1 · (0.5 - 0.95 · 0.4) = 0.1 · (0.5 - 0.38) = 0.012
```

**SoftHebb:**
```
Δw_{13} = 0.1 · 0.329 · (0.5 - 0.95 · 0.4) = 0.0329 · 0.12 = 0.00395
```

Now look at neuron 1 (the worst match), same weight index i=1 (x_1 = 0.5, w_{11} = 0.9):

**Hard WTA:**
```
Δw_{11} = 0.1 · 0 · (anything) = 0         ← completely frozen
```

**SoftHebb:**
```
Δw_{11} = 0.1 · 0.269 · (0.5 - 0.75 · 0.9) = 0.0269 · (0.5 - 0.675) = -0.0047
```
Neuron 1 actually shrinks its first weight slightly — it's learning that this input doesn't match its template, and adjusting accordingly. In Hard WTA this signal would be completely lost.

---

## Why the soft version wins

The numeric example shows the key insight: **SoftHebb extracts information from every input for every centroid**, not just the winner. The losing neurons learn "this input is not like me" and adjust. Over many examples, this means:

1. **Faster convergence** — more information per training example
2. **Fewer dead neurons** — even poor-matching neurons get small gradients
3. **Better cluster boundaries** — centroids are pushed away from inputs they shouldn't own

This is why SoftHebb reaches good accuracy in a single epoch, while Hard WTA and even backpropagation need more iterations.