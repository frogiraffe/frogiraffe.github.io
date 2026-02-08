---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/false-discovery-rate-fdr/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **False Discovery Rate (FDR)** is the expected proportion of ==false positives among all rejected hypotheses==.

$$\text{FDR} = E\left[\frac{V}{R}\right]$$

Where V = false positives, R = total rejections.

---

## vs FWER

| Control | What it limits |
|---------|---------------|
| **FWER** | P(≥1 false positive) |
| **FDR** | Expected % of false discoveries |

FDR is less stringent, more powerful when many tests are conducted.

---

## Benjamini-Hochberg Procedure

1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(m)}$
2. Find largest k where $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all $H_{(1)}, \dots, H_{(k)}$

---

## Python Implementation

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205]
reject, pvals_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print("Adjusted p-values:", pvals_adj)
print("Rejected:", reject)
```

---

## R Implementation

```r
p_values <- c(0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205)
p.adjust(p_values, method = "BH")
```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Multiple Comparisons Problem\|Multiple Comparisons Problem]] - The problem FDR addresses
- q-value (Storey) - Variation on FDR

---

## When to Use

> [!success] Use False Discovery Rate (FDR) When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Article:** Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289-300. [JSTOR](http://www.jstor.org/stable/2346101)
