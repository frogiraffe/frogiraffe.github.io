---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/hypergeometric-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Hypergeometric Distribution** models the number of ==successes in $n$ draws without replacement== from a finite population. Unlike the Binomial, probability changes after each draw.

![Hypergeometric Distribution showing PMF examples|500](https://upload.wikimedia.org/wikipedia/commons/c/c1/HypergeomProb.svg)
*Figure 1: Hypergeometric PMF for different parameter combinations.*

---

> [!tip] Intuition (ELI5): The Lottery Draw
> Imagine a bowl with 10 red and 90 white balls. You draw 5 balls *without putting them back*. The Hypergeometric tells you the probability of getting exactly 2 red balls. Unlike flipping coins, each draw changes what's left in the bowl.

---

## Purpose

1. Model **sampling without replacement** from finite populations
2. Quality control: Defectives in a batch sample
3. Foundation for [[30_Knowledge/Stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]]

---

## When to Use

> [!success] Use Hypergeometric Distribution When...
> - Sampling **without replacement**
> - Population is **finite** and relatively small
> - Probability changes as items are removed

---

## When NOT to Use

> [!danger] Do NOT Use Hypergeometric Distribution When...
> - **Sampling with replacement:** Use [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]
> - **Very large population:** Hypergeometric ≈ Binomial when $N >> n$
> - **Continuous outcomes:** Hypergeometric is for discrete counts
> - **Unknown population size:** Need to know $N$, $K$ exactly

---

## Theoretical Background

### Notation

$$
X \sim \text{Hypergeometric}(N, K, n)
$$

where:
- $N$ = total population size
- $K$ = number of success states in population
- $n$ = number of draws
- $X$ = number of observed successes

### Probability Mass Function (PMF)

$$
P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}
$$

**Understanding the Formula:**
- $\binom{K}{k}$: Ways to choose $k$ successes from $K$ total successes
- $\binom{N-K}{n-k}$: Ways to choose remaining draws from failures
- $\binom{N}{n}$: Total ways to draw $n$ items from $N$

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $\mu = n \cdot \frac{K}{N}$ |
| **Variance** | $\sigma^2 = n \cdot \frac{K}{N} \cdot \frac{N-K}{N} \cdot \frac{N-n}{N-1}$ |
| **Support** | $\max(0, n-N+K) \le k \le \min(n, K)$ |

### Finite Population Correction

The factor $\frac{N-n}{N-1}$ in the variance is called the **finite population correction (FPC)**. It reduces variance when sampling a large fraction of the population.

---

## Key Difference from Binomial

| Feature | Hypergeometric | Binomial |
|---------|----------------|----------|
| **Sampling** | Without replacement | With replacement |
| **Probability** | Changes each draw | Constant |
| **Population** | Finite, known | Infinite/large |
| **Independence** | Draws are dependent | Draws are independent |
| **Variance** | Lower (FPC factor) | Higher |

---

## Worked Example: Quality Control

> [!example] Problem
> A shipment contains **100 light bulbs**, of which **10 are defective**.
> An inspector randomly selects **15 bulbs** for testing.
> 
> **Questions:**
> 1. What is the probability of finding **exactly 2 defectives**?
> 2. What is the probability of finding **at least 1 defective**?

**Solution:**

Setup: $N=100$, $K=10$ (defectives), $n=15$ (sample)

**1. P(X = 2):**
$$P(X=2) = \frac{\binom{10}{2}\binom{90}{13}}{\binom{100}{15}}$$

**2. P(X ≥ 1):**
$$P(X \ge 1) = 1 - P(X=0) = 1 - \frac{\binom{10}{0}\binom{90}{15}}{\binom{100}{15}}$$

**Verification with Code:**
```python
from scipy.stats import hypergeom

N, K, n = 100, 10, 15
dist = hypergeom(M=N, n=K, N=n)  # scipy notation: M=pop, n=successes, N=draws

# P(X = 2)
print(f"P(X=2): {dist.pmf(2):.4f}")  # ~0.2743

# P(X >= 1)
print(f"P(X>=1): {1 - dist.pmf(0):.4f}")  # ~0.8095

# Expected defectives
print(f"E[X]: {dist.mean():.2f}")  # 1.50
```

---

## Assumptions

- [ ] **Finite population:** Population size $N$ is known and fixed.
  - *Example:* 52-card deck ✓ vs Infinite population ✗
  
- [ ] **Two categories:** Items are either success or failure.
  - *Example:* Defective/good ✓ vs 5 quality levels ✗
  
- [ ] **No replacement:** Drawn items are not returned.
  - *Example:* Card drawing ✓ vs Dice rolling ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **Small sample from large population:** When $n << N$, Hypergeometric ≈ Binomial. Use simpler Binomial.
> 2. **Parameter estimation:** Requires knowing $N$ and $K$ exactly—often unknown in practice.
> 3. **Computational complexity:** Factorials can overflow for large values.

---

## Python Implementation

```python
from scipy.stats import hypergeom
import numpy as np
import matplotlib.pyplot as plt

# Draw 5 cards, how many aces?
# N=52 (deck), K=4 (aces), n=5 (draws)
N, K, n = 52, 4, 5
dist = hypergeom(M=N, n=K, N=n)

# PMF
print(f"P(exactly 1 ace): {dist.pmf(1):.4f}")  # ~0.2995
print(f"P(at least 1 ace): {1 - dist.pmf(0):.4f}")  # ~0.3412

# Visualize
x = np.arange(0, min(K, n) + 1)
plt.bar(x, dist.pmf(x), alpha=0.7, edgecolor='black')
plt.xlabel('Number of Aces')
plt.ylabel('Probability')
plt.title(f'Hypergeometric(N={N}, K={K}, n={n})')
plt.xticks(x)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

**Expected Output:**
```
P(exactly 1 ace): 0.2995
P(at least 1 ace): 0.3412
```

---

## R Implementation

```r
# P(k=1 ace in 5 cards)
# dhyper(x, m, n, k): x=successes, m=success states, n=failures, k=draws
dhyper(1, m=4, n=48, k=5)  # 0.2995

# P(at least 1 ace)
1 - dhyper(0, m=4, n=48, k=5)  # 0.3412

# Mean
5 * (4/52)  # 0.385
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **$n << N$** | Hypergeometric ≈ Binomial; use simpler model |
| **$n$ close to $N$** | Strong FPC effect; variance much lower than Binomial |
| **All successes selected** | When $n \ge K$, P(draw all K) > 0 |
| **Comparing to Binomial** | Hypergeometric has *less* variance due to FPC |

---

## Applications

1. **Quality Control:** Defectives in sample from batch
2. **Card Games:** Specific hands (aces in poker)
3. **[[30_Knowledge/Stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]]:** Based on hypergeometric distribution
4. **Sampling Surveys:** When sampling fraction is large

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Sampling with replacement
- [[30_Knowledge/Stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]] - Uses hypergeometric for contingency tables
- [[30_Knowledge/Stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]] - >2 categories

### Applications
- [[30_Knowledge/Stats/01_Foundations/Sampling Bias\|Sampling Bias]] - Finite population considerations
- [[30_Knowledge/Stats/01_Foundations/Stratified Sampling\|Stratified Sampling]] - Often uses hypergeometric

### Other Related Topics

{ .block-language-dataview}

---

## References

1. Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. [Available online](https://www.pearson.com/us/higher-education/program/Ross-A-First-Course-in-Probability-9th-Edition/PGM220165.html)

2. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)
