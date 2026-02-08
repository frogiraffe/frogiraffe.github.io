---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/population-variance/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Population Variance** measures the average squared deviation from the population mean. Unlike [[30_Knowledge/Stats/01_Foundations/Sample Variance\|Sample Variance]], it divides by N, not N-1.

$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
$$

---

## Population vs Sample Variance

| Aspect | Population (σ²) | Sample (s²) |
|--------|-----------------|-------------|
| Divisor | N | n-1 |
| When used | Full population known | Estimating from sample |
| Bias | N/A | Unbiased estimator |

---

## Python Implementation

```python
import numpy as np

data = np.array([4, 8, 6, 5, 3, 2, 8, 9, 2, 5])

pop_var = np.var(data, ddof=0)    # Population (ddof=0)
sample_var = np.var(data, ddof=1)  # Sample (ddof=1)

print(f"Population Variance: {pop_var:.2f}")
print(f"Sample Variance: {sample_var:.2f}")
```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Sample Variance\|Sample Variance]] — Uses n-1 (Bessel's correction)
- [[30_Knowledge/Stats/01_Foundations/Standard Deviation\|Standard Deviation]] — Square root of variance

---

## When to Use

> [!success] Use Population Variance When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## R Implementation

```r
# Population Variance in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** DeGroot, M. H. (2012). *Probability and Statistics*. Pearson.
