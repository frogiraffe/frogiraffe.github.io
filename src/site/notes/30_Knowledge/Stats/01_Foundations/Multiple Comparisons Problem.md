---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/multiple-comparisons-problem/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> The **Multiple Comparisons Problem** occurs when performing many hypothesis tests ==inflates the Type I error rate== beyond the nominal α level.

![Quantile-Quantile plot illustrating Multiple Comparisons behavior](https://commons.wikimedia.org/wiki/Special:FilePath/Quantile_meta_test.svg)

With m tests at α = 0.05:
$$P(\text{at least 1 false positive}) = 1 - (1 - \alpha)^m$$

---

## Example

| # Tests | P(≥1 False Positive) |
|---------|---------------------|
| 1 | 5% |
| 10 | 40% |
| 100 | 99.4% |

---

## Correction Methods

| Method | Control | Formula |
|--------|---------|---------|
| **Bonferroni** | FWER | $\alpha' = \alpha / m$ |
| **Holm** | FWER (less conservative) | Step-down |
| **Benjamini-Hochberg** | FDR | Ranks p-values |

---

## Python Implementation

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.01, 0.03, 0.04, 0.08, 0.15]

# Bonferroni
reject_bonf, pvals_bonf, _, _ = multipletests(p_values, method='bonferroni')

# Benjamini-Hochberg (FDR)
reject_bh, pvals_bh, _, _ = multipletests(p_values, method='fdr_bh')
```

---

## R Implementation

```r
p_values <- c(0.01, 0.03, 0.04, 0.08, 0.15)

p.adjust(p_values, method = "bonferroni")
p.adjust(p_values, method = "BH")  # Benjamini-Hochberg
```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/False Discovery Rate (FDR)\|False Discovery Rate (FDR)]] - Alternative control target
- [[30_Knowledge/Stats/02_Statistical_Inference/Tukey's HSD\|Tukey's HSD]] - Specific to ANOVA

---

## When to Use

> [!success] Use Multiple Comparisons Problem When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Article:** Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289-300. [JSTOR Link](http://www.jstor.org/stable/2346101)
