---
{"dg-publish":true,"permalink":"/stats/01-foundations/quantiles-and-quartiles/","tags":["Descriptive-Statistics","Statistics"]}
---


## Definition

> [!abstract] Core Statement
> **Quantiles** divide data into equal-sized groups based on rank. **Quartiles** specifically divide data into four parts (Q1, Q2, Q3).

---

## Key Quantiles

| Quantile | Name | Interpretation |
|----------|------|----------------|
| Q1 (25th) | First Quartile | 25% of data below |
| Q2 (50th) | [[stats/01_Foundations/Median\|Median]] | 50% below |
| Q3 (75th) | Third Quartile | 75% below |
| P90 | 90th Percentile | Top 10% |

---

## IQR (Interquartile Range)

$$
\text{IQR} = Q3 - Q1
$$

Used for:
- **Outlier detection**: Points outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$
- **Box plots**: The "box" spans Q1 to Q3

---

## Python Implementation

```python
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])

Q1 = np.percentile(data, 25)
Q2 = np.percentile(data, 50)  # Median
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

print(f"Q1: {Q1}, Median: {Q2}, Q3: {Q3}")
print(f"IQR: {IQR}")

# Outlier bounds
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f"Outliers outside: [{lower}, {upper}]")
```

---

## R Implementation

```r
quantile(data, probs = c(0.25, 0.5, 0.75))
IQR(data)
boxplot(data)
```

---

## Related Concepts

- [[stats/01_Foundations/Median\|Median]] — Q2 = 50th percentile
- [[Box Plots\|Box Plots]] — Visualizes quartiles

---

## References

- **Book:** Hoaglin, D. C., et al. (2000). *Understanding Robust and Exploratory Data Analysis*. Wiley.
