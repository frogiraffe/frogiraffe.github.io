---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/kendall-tau/","tags":["probability","foundations"]}
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

- [[30_Knowledge/Stats/01_Foundations/Spearman Rank Correlation\|Spearman Rank Correlation]] — Alternative rank correlation
- [[30_Knowledge/Stats/01_Foundations/Correlation Analysis\|Correlation Analysis]] — Overview

---

## When to Use

> [!success] Use Kendall Tau When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Paper:** Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*, 30(1-2), 81-93.
