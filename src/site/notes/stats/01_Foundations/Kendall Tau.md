---
{"dg-publish":true,"permalink":"/stats/01-foundations/kendall-tau/","tags":["Statistics","Correlation","Non-Parametric"]}
---


## Definition

> [!abstract] Core Statement
> **Kendall Tau** measures ==concordance between rankings==. It counts concordant vs discordant pairs, making it more robust than Spearman for small samples or many ties.

$$
\tau = \frac{\text{(concordant pairs)} - \text{(discordant pairs)}}{\binom{n}{2}}
$$

---

## Kendall vs Spearman

| Aspect | Spearman | Kendall |
|--------|----------|---------|
| Based on | Rank differences | Pair concordance |
| Ties | Less robust | More robust |
| Interpretation | Similar to Pearson | Probability-based |

---

## Python Implementation

```python
from scipy import stats

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

tau, p = stats.kendalltau(x, y)
print(f"Kendall τ = {tau:.3f}, p = {p:.4f}")
```

---

## R Implementation

```r
cor.test(x, y, method = "kendall")
```

---

## Related Concepts

- [[stats/01_Foundations/Spearman Rank Correlation\|Spearman Rank Correlation]] — Alternative rank correlation
- [[stats/01_Foundations/Correlation Analysis\|Correlation Analysis]] — Overview

---

## References

- **Paper:** Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*, 30(1-2), 81-93.
