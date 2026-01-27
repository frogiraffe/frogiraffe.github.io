---
{"dg-publish":true,"permalink":"/stats/01-foundations/sample-variance/","tags":["Descriptive-Statistics","Variability"]}
---


## Definition

> [!abstract] Core Statement
> **Sample Variance** measures the ==average squared deviation from the sample mean==, using n-1 (Bessel's correction) for unbiased estimation of population variance.

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

---

## Why n-1?

- Sample mean is estimated from data (loss of 1 df)
- Using n would underestimate population variance
- Known as **Bessel's correction**

---

## Properties

| Property | Value |
|----------|-------|
| **Unbiased** | $E[s^2] = \sigma^2$ |
| **Units** | Squared units of data |
| **Always** | ≥ 0 |

---

## Python Implementation

```python
import numpy as np

data = np.array([4, 8, 6, 5, 3, 2, 8, 9, 2, 5])
sample_var = np.var(data, ddof=1)  # ddof=1 for sample variance
print(f"Sample variance: {sample_var:.2f}")
```

---

## R Implementation

```r
data <- c(4, 8, 6, 5, 3, 2, 8, 9, 2, 5)
var(data)  # Uses n-1 by default
```

---

## Related Concepts

- [[stats/01_Foundations/Standard Deviation\|Standard Deviation]] - Square root of variance
- [[Population Variance\|Population Variance]] - Uses n instead of n-1

---

## References

- **Book:** DeGroot, M. H. (2012). *Probability and Statistics*. Pearson. [Pearson Link](https://www.pearson.com/us/higher-education/program/De-Groot-Probability-and-Statistics-4th-Edition/PGM248744.html)
