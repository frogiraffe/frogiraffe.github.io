---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/median/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> The **Median** is the ==middle value== when data is ordered. It divides the distribution in half — 50% of values are below, 50% above. It's robust to outliers unlike the mean.

---

> [!tip] Intuition (ELI5): The Middle Kid
> Line up all your data points by size. The median is the kid standing exactly in the middle. If there's an even number, take the average of the two middle kids. The median doesn't care if the tallest kid is 6 feet or 100 feet—it's still in the middle.

---

## Calculation

**Odd n:** Middle value
**Even n:** Average of two middle values

$$
\text{Median} = \begin{cases} 
x_{(n+1)/2} & \text{if } n \text{ is odd} \\
\frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ is even}
\end{cases}
$$

---

## Python Implementation

```python
import numpy as np
from scipy import stats

data = np.array([10, 20, 30, 40, 1000])

median = np.median(data)
mean = np.mean(data)

print(f"Median: {median}")  # 30 (robust)
print(f"Mean: {mean}")      # 220 (pulled by outlier)
```

---

## R Implementation

```r
data <- c(10, 20, 30, 40, 1000)
median(data)  # 30
mean(data)    # 220
```

---

## When to Use Median

| Scenario | Use Median |
|----------|------------|
| Income/wealth data | ✓ |
| Data with outliers | ✓ |
| Skewed distributions | ✓ |
| Ordinal data | ✓ |

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Mean\|Mean]] — Sensitive to outliers
- [[30_Knowledge/Stats/01_Foundations/Skewness\|Skewness]] — Mean vs Median difference
- [[30_Knowledge/Stats/01_Foundations/Quantiles and Quartiles\|Quantiles and Quartiles]] — Median = 50th percentile

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Wackerly, D., Mendenhall, W., & Scheaffer, R. (2014). *Mathematical Statistics with Applications*.
