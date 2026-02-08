---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/spearman-rank-correlation/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Spearman Rank Correlation** measures ==monotonic relationships== by computing Pearson correlation on the ranks of the data, making it robust to outliers and non-linearity.

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

Where $d_i$ = difference between ranks of paired observations.

---

## Spearman vs Pearson

| Aspect | Pearson | Spearman |
|--------|---------|----------|
| **Measures** | Linear relationship | Monotonic relationship |
| **Assumes** | Normality, interval data | Ordinal data OK |
| **Outliers** | Sensitive | Robust |

---

## Python Implementation

```python
from scipy import stats

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]  # Quadratic but monotonic

rho, p = stats.spearmanr(x, y)
print(f"Spearman ρ = {rho:.3f}, p = {p:.4f}")

# Compare with Pearson
r, _ = stats.pearsonr(x, y)
print(f"Pearson r = {r:.3f}")  # Lower due to non-linearity
```

---

## R Implementation

```r
cor(x, y, method = "spearman")
cor.test(x, y, method = "spearman")
```

---

## When to Use

> [!success] Use Spearman When...
> - Data is ordinal (rankings, Likert scales)
> - Relationship is monotonic but not linear
> - Outliers are present

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Correlation Analysis\|Correlation Analysis]] — Overview of correlation
- [[30_Knowledge/Stats/01_Foundations/Kendall Tau\|Kendall Tau]] — Alternative rank correlation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.
