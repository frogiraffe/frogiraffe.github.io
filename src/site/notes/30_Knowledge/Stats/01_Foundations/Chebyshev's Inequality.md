---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/chebyshev-s-inequality/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Chebyshev's Inequality** provides an upper bound on the probability that a random variable deviates from its mean by more than $k$ standard deviations. It applies to ==any distribution== with finite mean and variance.

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

Equivalently:
$$P(|X - \mu| < k\sigma) \geq 1 - \frac{1}{k^2}$$

---

## Purpose

1.  Bound tail probabilities **without knowing the distribution**.
2.  Provide worst-case guarantees for outlier detection.
3.  Prove the **Law of Large Numbers** (weak version).
4.  Justify robustness of statistical estimators.

---

## When to Use

> [!success] Use Chebyshev's Inequality When...
> - The distribution is **unknown** or **non-normal**.
> - You need a **conservative bound** (worst-case scenario).
> - Proving theoretical results about convergence.

> [!failure] Limitations
> - The bound is often **very loose** for well-behaved distributions.
> - For normal distributions, use the 68-95-99.7 rule instead.
> - Only useful when you know $\mu$ and $\sigma$.

---

## Theoretical Background

### The Bound

| $k$ (std devs) | Chebyshev Bound | Normal (actual) |
|----------------|-----------------|-----------------|
| 1 | ≤ 100% | 31.7% |
| 2 | ≤ 25% | 4.6% |
| 3 | ≤ 11.1% | 0.3% |
| 4 | ≤ 6.25% | 0.006% |

> [!important] Key Insight
> Chebyshev is **distribution-free** but **conservative**. For normal data, actual tail probabilities are much smaller than the bound.

### One-Sided Version

$$P(X - \mu \geq k\sigma) \leq \frac{1}{1 + k^2}$$

### Proof Sketch

Using Markov's inequality on $(X - \mu)^2$:
$$P(|X - \mu| \geq k\sigma) = P((X - \mu)^2 \geq k^2\sigma^2) \leq \frac{E[(X-\mu)^2]}{k^2\sigma^2} = \frac{\sigma^2}{k^2\sigma^2} = \frac{1}{k^2}$$

---

## Worked Example

> [!example] Problem
> A factory produces bolts with mean length 10 cm and standard deviation 0.5 cm. The distribution is unknown.
> 
> What is the maximum probability that a randomly selected bolt differs from 10 cm by more than 1.5 cm?

**Solution:**

1. **Identify parameters:**
   - $\mu = 10$ cm
   - $\sigma = 0.5$ cm
   - Deviation = 1.5 cm = $k \times \sigma$
   - $k = 1.5 / 0.5 = 3$

2. **Apply Chebyshev:**
$$P(|X - 10| \geq 1.5) \leq \frac{1}{3^2} = \frac{1}{9} \approx 0.111$$

**Answer:** At most **11.1%** of bolts are outside the range [8.5, 11.5] cm.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

def chebyshev_bound(k):
    """Chebyshev upper bound for k standard deviations."""
    return 1 / k**2

def compare_bounds(k_values, dist='norm'):
    """Compare Chebyshev bound with actual tail probability."""
    print(f"{'k':>4} | {'Chebyshev':>10} | {'Actual':>10} | {'Ratio':>8}")
    print("-" * 40)
    for k in k_values:
        cheb = chebyshev_bound(k)
        if dist == 'norm':
            actual = 2 * (1 - stats.norm.cdf(k))  # Two-tailed
        else:
            actual = cheb  # Unknown
        ratio = cheb / actual if actual > 0 else float('inf')
        print(f"{k:>4} | {cheb:>10.4f} | {actual:>10.4f} | {ratio:>8.1f}x")

# Compare for normal distribution
compare_bounds([1, 2, 3, 4, 5])
# Output shows Chebyshev is 3-100x looser than actual

# Verify with simulation
np.random.seed(42)
data = np.random.normal(0, 1, 100000)
k = 2
empirical = np.mean(np.abs(data) >= k)
print(f"\nEmpirical P(|X| >= {k}): {empirical:.4f}")
print(f"Chebyshev bound: {chebyshev_bound(k):.4f}")
print(f"Normal actual: {2*(1-stats.norm.cdf(k)):.4f}")
```

---

## R Implementation

```r
chebyshev_bound <- function(k) {
  1 / k^2
}

# Compare bounds
k_values <- c(1, 2, 3, 4, 5)
actual_normal <- 2 * pnorm(-k_values)

data.frame(
  k = k_values,
  Chebyshev = chebyshev_bound(k_values),
  Normal_Actual = actual_normal,
  Ratio = chebyshev_bound(k_values) / actual_normal
)

# Simulation verification
set.seed(42)
data <- rnorm(100000)
k <- 2
empirical <- mean(abs(data) >= k)
cat("Empirical:", empirical, "\n")
cat("Chebyshev:", chebyshev_bound(k), "\n")
```

---

## ML Applications

| Application | How Chebyshev Is Used |
|-------------|----------------------|
| **Outlier Detection** | Flag points > k std devs as anomalies (conservative). |
| **Robust Statistics** | Justify trimmed means, winsorization. |
| **Confidence Bounds** | Distribution-free confidence intervals. |
| **PAC Learning** | Bound generalization error. |
| **Streaming Algorithms** | Bound estimation errors with limited data. |

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **k = 2, bound = 25%** | At most 25% of data is > 2σ from mean. |
| **Bound >> actual** | Distribution is well-behaved (use tighter bounds). |
| **Need exact P** | Chebyshev insufficient; identify distribution. |

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Variance\|Variance]] - Chebyshev uses variance directly.
- [[30_Knowledge/Stats/01_Foundations/Standard Deviation\|Standard Deviation]] - Deviation measured in σ units.
- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] - Has much tighter bounds (68-95-99.7 rule).
- [[30_Knowledge/Stats/01_Foundations/Law of Large Numbers\|Law of Large Numbers]] - Proven using Chebyshev.
- [[30_Knowledge/Stats/01_Foundations/Markov Chains\|Markov Chains]] - Markov inequality is the parent result.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. Chapter 3.
- **Book:** Ross, S. M. (2019). *A First Course in Probability* (10th ed.). Pearson.
