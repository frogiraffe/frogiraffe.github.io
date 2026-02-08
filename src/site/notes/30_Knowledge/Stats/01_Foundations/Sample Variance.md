---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/sample-variance/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Sample Variance** measures the ==average squared deviation from the sample mean==, using n-1 (Bessel's correction) for unbiased estimation of population variance.

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

---

> [!tip] Intuition (ELI5)
> When you calculate the mean from your sample, you've already "used up" one piece of independent information. You only have n-1 pieces of truly free information left to estimate spread.

---

## Why n-1? (Bessel's Correction)

```
Population σ² = 10 (true variance)

With n (biased):
┌────────────────────────────────────────────┐
│ ■■■■■■■■■■ 8.5  ← Underestimates!          │
└────────────────────────────────────────────┘

With n-1 (unbiased):
┌────────────────────────────────────────────┐
│ ■■■■■■■■■■■■ 10.0  ← Hits target on average│
└────────────────────────────────────────────┘
```

### Mathematical Reason
- Using $\bar{x}$ instead of $\mu$ constrains the deviations
- Deviations must sum to zero: $\sum(x_i - \bar{x}) = 0$
- Only n-1 values are "free" to vary

---

## Python Demonstration

```python
import numpy as np

np.random.seed(42)

# Population with known variance
population = np.random.normal(loc=50, scale=10, size=100000)
true_variance = population.var()  # ≈ 100

# ========== SIMULATION: BIAS OF n VS n-1 ==========
n_samples = 1000
sample_size = 10

biased_estimates = []
unbiased_estimates = []

for _ in range(n_samples):
    sample = np.random.choice(population, size=sample_size)
    biased_estimates.append(sample.var(ddof=0))    # n
    unbiased_estimates.append(sample.var(ddof=1))  # n-1

print(f"True variance: {true_variance:.2f}")
print(f"Mean of biased (n): {np.mean(biased_estimates):.2f}")      # Underestimates!
print(f"Mean of unbiased (n-1): {np.mean(unbiased_estimates):.2f}")  # ≈ true

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.hist(biased_estimates, bins=30, alpha=0.5, label='n (biased)')
plt.hist(unbiased_estimates, bins=30, alpha=0.5, label='n-1 (unbiased)')
plt.axvline(true_variance, color='red', linestyle='--', label='True σ²')
plt.legend()
plt.title("Bessel's Correction Effect")
plt.show()
```

---

## Properties

| Property | Value |
|----------|-------|
| **Unbiased** | $E[s^2] = \sigma^2$ |
| **Units** | Squared units of data |
| **Range** | Always ≥ 0 |
| **Degrees of Freedom** | n - 1 |

---

## Worked Example

Given data: **4, 8, 6, 5, 3**

1. Mean: $\bar{x} = (4+8+6+5+3)/5 = 5.2$
2. Squared deviations: $(4-5.2)^2 + (8-5.2)^2 + ... = 14.8$
3. Sample variance: $s^2 = 14.8 / (5-1) = 3.7$

```python
data = np.array([4, 8, 6, 5, 3])
print(np.var(data, ddof=1))  # 3.7
```

---

## R Implementation

```r
data <- c(4, 8, 6, 5, 3, 2, 8, 9, 2, 5)
var(data)  # Uses n-1 by default (unbiased)

# Population variance (if data IS the population)
pop_var <- sum((data - mean(data))^2) / length(data)
```

---

## Common Pitfall

> [!warning] NumPy Default
> `np.var()` uses n (biased) by default!
> ```python
> np.var(data)        # Uses n (WRONG for samples)
> np.var(data, ddof=1) # Uses n-1 (CORRECT)
> ```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Standard Deviation\|Standard Deviation]] — Square root of variance
- [[30_Knowledge/Stats/01_Foundations/Population Variance\|Population Variance]] — Uses n instead of n-1
- [[30_Knowledge/Stats/02_Statistical_Inference/Degrees of Freedom\|Degrees of Freedom]] — Why we lose one

---

## When to Use

> [!success] Use Sample Variance When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Sample Variance
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## References

- **Book:** DeGroot, M. H. (2012). *Probability and Statistics*. Pearson.
- **Historical:** Bessel, F. W. (1818). Fundamenta Astronomiae.

