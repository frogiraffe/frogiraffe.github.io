---
{"dg-publish":true,"permalink":"/stats/01-foundations/population-variance/","tags":["Statistics","Variability"]}
---


## Definition

> [!abstract] Core Statement
> **Population Variance** measures the average squared deviation from the population mean. Unlike [[stats/01_Foundations/Sample Variance\|Sample Variance]], it divides by N, not N-1.

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

- [[stats/01_Foundations/Sample Variance\|Sample Variance]] — Uses n-1 (Bessel's correction)
- [[stats/01_Foundations/Standard Deviation\|Standard Deviation]] — Square root of variance

---

## References

- **Book:** DeGroot, M. H. (2012). *Probability and Statistics*. Pearson.
